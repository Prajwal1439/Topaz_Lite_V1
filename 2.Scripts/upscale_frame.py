import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_image(input_path, output_path, scalingFactor=2, model_path=None):
    # Load model only once (use global variable)
    global global_upsampler
    if 'global_upsampler' not in globals():
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "1.Models", "RealESRGAN_x2plus.pth")

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=2
        )
        global_upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=400,  # Increased tile size for better CPU utilization
            tile_pad=20,
            pre_pad=0,
            half=False,
            device=torch.device('cpu')
        )

    # Process image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR) if isinstance(input_path, str) else input_path
    if img is None:
        print(f"[❌] Could not load input image: {input_path}")
        return

    try:
        with torch.no_grad():  # Disable gradient calculation
            if scalingFactor == 2:
                output, _ = global_upsampler.enhance(img, outscale=2)
            else:
                output, _ = global_upsampler.enhance(img, outscale=1.5)

        cv2.imwrite(output_path, output)
    except Exception as e:
        print(f"[❌] Error processing {input_path}: {str(e)}")
