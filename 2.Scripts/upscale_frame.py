import os
import cv2
import torch
import numpy as np
import multiprocessing
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import onnxruntime as ort

# Global model instances
global_upsampler_quality = None
global_upsampler_fast = None
onnx_model_input_size = 128

def initialize_fast_model(model_path=None):
    """Optimized multi-core ONNX initialization"""
    global onnx_model_input_size

    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "1.Models", "Real-ESRGAN-General-x4v3.onnx")

    # Configure for maximum CPU utilization
    sess_options = ort.SessionOptions()
    num_cores = max(1, multiprocessing.cpu_count())

    sess_options.intra_op_num_threads = num_cores
    sess_options.inter_op_num_threads = 2
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ('CPUExecutionProvider', {
            'intra_op_num_threads': num_cores,
            'inter_op_num_threads': 2,
            'arena_extend_strategy': 'kSameAsRequested'
        })
    ]

    try:
        session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=sess_options
        )
        input_shape = session.get_inputs()[0].shape
        onnx_model_input_size = input_shape[2]
        print(f"[ℹ️] ONNX Model initialized with {num_cores} CPU cores")
        return session
    except Exception as e:
        raise RuntimeError(f"ONNX init failed: {str(e)}")

def validate_image(img):
    """Ensure image is valid before processing"""
    if img is None:
        raise ValueError("Image is None")
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(img)}")
    if img.size == 0:
        raise ValueError("Empty image array")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

def process_with_fast_model(img, session, scalingFactor):
    """
    Optimized version with enhanced error handling and performance
    """
    try:
        # Validate input
        validate_image(img)
        h, w = img.shape[:2]

        # ============================================================
        # 1. Model Configuration
        # ============================================================
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        model_scale = output_shape[2] // input_shape[2]
        tile_size = input_shape[2]

        # ============================================================
        # 2. Smart Scaling Strategy
        # ============================================================
        # Calculate working scale and final dimensions
        working_scale = model_scale
        final_h, final_w = int(h * scalingFactor), int(w * scalingFactor)

        # ============================================================
        # 3. Padding Calculation
        # ============================================================
        def calculate_padding(dim, multiple):
            return (multiple - (dim % multiple)) % multiple

        pad_w = calculate_padding(w, tile_size)
        pad_h = calculate_padding(h, tile_size)

        # ============================================================
        # 4. Image Preparation
        # ============================================================
        if pad_w > 0 or pad_h > 0:
            img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=pad_h,
                left=0,
                right=pad_w,
                borderType=cv2.BORDER_REFLECT101
            )

        # Convert to RGB float32
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ============================================================
        # 5. Tile Processing
        # ============================================================
        output = np.zeros(
            (int(img.shape[0] * working_scale),
             int(img.shape[1] * working_scale), 3),
            dtype=np.float32
        )

        for y in range(0, img.shape[0], tile_size):
            for x in range(0, img.shape[1], tile_size):
                # Extract tile with bounds checking
                tile = img_rgb[y:y + tile_size, x:x + tile_size]

                # Process tile
                tile_tensor = np.transpose(tile, (2, 0, 1))[np.newaxis, ...]
                result = session.run(None, {input_name: tile_tensor})[0]
                result = np.squeeze(result, 0).transpose(1, 2, 0)

                # Calculate output position
                out_y, out_x = y * working_scale, x * working_scale
                out_size = tile_size * working_scale

                # Write to output
                output[out_y:out_y + out_size, out_x:out_x + out_size] = result

        # ============================================================
        # 6. Final Processing
        # ============================================================
        # Remove padding
        output = output[:h * working_scale, :w * working_scale]

        # Resize to target scale if needed
        if working_scale != scalingFactor:
            output = cv2.resize(
                output,
                (final_w, final_h),
                interpolation=cv2.INTER_LANCZOS4 if scalingFactor < working_scale else cv2.INTER_CUBIC
            )

        # Convert to BGR uint8
        output = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).clip(0, 255).astype(np.uint8)

        return output

    except Exception as e:
        print(f"[❌] Processing error: {str(e)}")
        raise

def upscale_image(input_path, output_path, scalingFactor=2, model_type="quality", model_path=None):
    """Main function with enhanced error handling"""
    global global_upsampler_quality, global_upsampler_fast

    # Load image with validation
    try:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR) if isinstance(input_path, str) else input_path
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")

        validate_image(img)

        if model_type == "quality":
            global global_upsampler_quality
            if global_upsampler_quality is None:  # Changed this condition
                if model_path is None:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, "..", "1.Models", "RealESRGAN_x2plus.pth")
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Quality model not found at: {model_path}")

                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23,
                    num_grow_ch=32, scale=2
                )
                global_upsampler_quality = RealESRGANer(  # Changed variable name here
                    scale=2,
                    model_path=model_path,
                    model=model,
                    tile=400,
                    tile_pad=20,
                    pre_pad=0,
                    half=False,
                    device=torch.device('cpu')
                )

            # Process image
            try:
                with torch.no_grad():
                    if scalingFactor == 2:
                        output, _ = global_upsampler_quality.enhance(img, outscale=2)  # Changed variable name here
                    else:
                        output, _ = global_upsampler_quality.enhance(img, outscale=1.5)  # Changed variable name here

                cv2.imwrite(output_path, output)
            except Exception as e:
                print(f"[❌] Error processing {input_path}: {str(e)}")
                raise
        else:
            if global_upsampler_fast is None:
                global_upsampler_fast = initialize_fast_model()
            output = process_with_fast_model(img, global_upsampler_fast, scalingFactor)
            cv2.imwrite(output_path, output)

        print(f"[✅] Successfully processed: {output_path}")

    except Exception as e:
        print(f"[❌] Critical error in upscale_image: {str(e)}")
        raise