import os
import shutil
import subprocess
import cv2
import gc
import psutil
from functools import partial
from tqdm import tqdm
from upscale_frame import upscale_image
from video_output import finalize_output
import datetime
from pathlib import Path
import torch
import multiprocessing

# Configure PyTorch to use all available threads
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

def cleanup_temp_dirs(temp_dir):
    """Remove temporary directories and their contents"""
    try:
        if os.path.exists(temp_dir):
            print(f"[🧹] Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[⚠️] Warning: Failed to clean up {temp_dir}: {str(e)}")

def init_worker(model_type):
    """Initialize worker process with proper resource limits"""
    # Configure based on model type
    if model_type == "quality":
        # Use all available threads for quality model
        torch.set_num_threads(multiprocessing.cpu_count())
    else:
        # Limit threads for fast model
        torch.set_num_threads(1)
    gc.collect()

def clean_memory():
    """Force clean memory and cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_system_resources(use_all_cores=True, model_type="quality"):
    """Calculate optimal number of workers based on system resources"""
    total_cores = multiprocessing.cpu_count()
    available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB

    # Adjust memory per worker based on model type
    mem_per_worker = 4.0 if model_type == "quality" else 2.5  # GB

    mem_based_workers = int(available_mem / mem_per_worker)

    if use_all_cores:
        core_based_workers = total_cores
    else:
        core_based_workers = max(1, total_cores - 1)

    return max(1, min(core_based_workers, mem_based_workers))

def extract_frames(video_path, frame_dir):
    """Optimized frame extraction with hardware acceleration"""
    os.makedirs(frame_dir, exist_ok=True)
    input_path = os.path.abspath(video_path).replace("\\", "/")
    output_path = os.path.abspath(os.path.join(frame_dir, "frame_%06d.png")).replace("\\", "/")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 30
        print(f"[⚠️] Couldn't determine FPS, defaulting to {fps}")

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        output_path
    ]

    print(f"[🔁] Extracting {total_frames} frames at {fps} FPS")
    try:
        subprocess.run(cmd, check=True)
        return fps
    except subprocess.CalledProcessError as e:
        print(f"[❌] FFmpeg failed with code {e.returncode}")
        raise e
    except FileNotFoundError:
        print("[❌] FFmpeg not found. Install with: sudo apt install ffmpeg")
        raise

def process_single_frame(file_info, enhanced_dir, model_type="quality"):
    """Process a single frame with model selection"""
    file, input_path = file_info
    output_path = os.path.abspath(os.path.join(enhanced_dir, file))

    try:
        if not os.path.isfile(input_path):
            print(f"[⚠️] Missing input frame: {input_path}")
            return False

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[⚠️] Failed to decode image: {input_path}")
            return False

        h, w = img.shape[:2]
        min_dim = min(h, w)

        # Dynamic scaling based on input resolution
        if min_dim >= 1080:
            scale = 720 / min_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaling = 1.5
        elif min_dim == 720:
            scaling = 1.5
        else:
            scaling = 2

        upscale_image(img, output_path, scaling, model_type=model_type)
        return True
    except Exception as e:
        print(f"[❌] Error processing {file}: {str(e)}")
        return False
    finally:
        if 'img' in locals():
            del img
        clean_memory()

def process_frame_wrapper(args):
    """Wrapper for frame processing with model type"""
    try:
        file_name, input_path, enhanced_dir, model_type = args
        if model_type == "quality":
            torch.set_num_threads(max(1, multiprocessing.cpu_count()))
        return process_single_frame((file_name, input_path), enhanced_dir, model_type)
    except Exception as e:
        print(f"[⚠️] Error in wrapper for {args[0]}: {str(e)}")
        return False


def enhance_frames(frame_dir, enhanced_dir, use_all_cores=True, model_type="quality"):
    """Enhanced frame processing with multiprocessing and model selection"""
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    frame_files = sorted([
        (f, os.path.join(frame_dir, f))
        for f in os.listdir(frame_dir)
        if f.lower().endswith('.png') and os.path.isfile(os.path.join(frame_dir, f))
    ], key=lambda x: int(''.join(filter(str.isdigit, x[0]))))

    if not frame_files:
        raise ValueError(f"No valid PNG frames found in: {frame_dir}")

    os.makedirs(enhanced_dir, exist_ok=True)
    num_workers = get_system_resources(use_all_cores)

    print(f"[ℹ️] Processing {len(frame_files)} frames with {num_workers} workers ({model_type} model)")

    args = [
        (file_name, file_path, os.path.abspath(enhanced_dir), model_type)
        for file_name, file_path in frame_files
    ]

    try:
        init_with_model = partial(init_worker, model_type=model_type)
        with multiprocessing.Pool(
                processes=num_workers,
                initializer=init_with_model,
                maxtasksperchild=20
        ) as pool:
            results = []
            with tqdm(total=len(frame_files), desc="[🔁] Enhancing frames") as pbar:
                for result in pool.imap_unordered(
                        process_frame_wrapper,
                        args,
                        chunksize=max(1, len(frame_files) // (num_workers * 2))
                ):
                    results.append(result)
                    pbar.update()

            success_count = sum(results)
            print(f"[✅] Successfully processed {success_count}/{len(frame_files)} frames")

    except Exception as e:
        print(f"[❌] Frame processing failed: {str(e)}")
        raise


def assemble_video(enhanced_dir, output_path, fps, codec='h264'):
    """Assemble enhanced frames into a video"""
    if not shutil.which('ffmpeg'):
        raise RuntimeError("FFmpeg not found. Install with: sudo apt install ffmpeg")

    if not os.path.exists(enhanced_dir):
        raise RuntimeError(f"Directory not found: {enhanced_dir}")

    try:
        frame_files = sorted(
            [f for f in os.listdir(enhanced_dir)
             if f.lower().endswith('.png') and os.path.isfile(os.path.join(enhanced_dir, f))],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )
    except ValueError:
        raise RuntimeError("Frame files must be numerically numbered (e.g., frame_0001.png)")

    if not frame_files:
        raise RuntimeError(f"No PNG frames found in: {enhanced_dir}")

    codec_config = {
        'h264': {
            'args': ['-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p'],
            'ext': '.mp4'
        },
        'h265': {
            'args': ['-c:v', 'libx265', '-crf', '20', '-preset', 'slow', '-pix_fmt', 'yuv420p10le'],
            'ext': '.mp4'
        },
        'prores': {
            'args': ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le'],
            'ext': '.mov'
        }
    }

    if codec not in codec_config:
        raise RuntimeError(f"Unsupported codec: {codec}")

    config = codec_config[codec]
    output_path = os.path.splitext(output_path)[0] + config['ext']

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(enhanced_dir, 'frame_%06d.png'),
        *config['args'],
        '-an',
        '-movflags', '+faststart',
        '-hide_banner',
        '-loglevel', 'error',
        output_path
    ]

    try:
        print(f"[⏳] Starting {codec} encode for {len(frame_files)} frames")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg failed to create output file")
        if os.path.getsize(output_path) == 0:
            raise RuntimeError("Output file is empty (0 bytes)")

        print(f"[✅] Success: {output_path} ({os.path.getsize(output_path) / 1024 ** 2:.1f}MB)")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = [
            "FFmpeg encoding failed!",
            f"Command: {' '.join(cmd)}",
            f"Exit code: {e.returncode}",
            "Error output:",
            e.stderr.strip()
        ]
        raise RuntimeError('\n'.join(error_msg))
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

def process_video(video_path, temp_dir='temp', resolution='720p', codec='h264', use_all_cores=True, model_type="quality"):
    """Main video processing pipeline"""
    cleanup_temp_dirs(temp_dir)
    clean_memory()

    try:
        frame_dir = os.path.join(temp_dir, 'frames')
        enhanced_dir = os.path.join(temp_dir, 'enhanced_frames')

        original_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = f"endResult_{original_name}_{resolution}_{codec}_{timestamp}"

        fps = extract_frames(video_path, frame_dir)
        enhance_frames(frame_dir, enhanced_dir, use_all_cores, model_type)

        intermediate_path = assemble_video(
            enhanced_dir,
            os.path.join(temp_dir, f"intermediate_{codec}"),
            fps,
            codec
        )

        final_output_path = finalize_output(
            intermediate_path,
            final_output,
            video_path,
            resolution,
            codec
        )

        print(f"[✅] Final output: {os.path.basename(final_output_path)}")
        print(f"     Saved to: {os.path.dirname(final_output_path)}")
        return final_output_path

    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}")
    finally:
        cleanup_temp_dirs(temp_dir)
        clean_memory()

if __name__ == '__main__':
    process_video(
        video_path='3.Test_Images_Or_Videos/sample_input.mp4',
        temp_dir='temp_processing',
        resolution='720p',
        codec='h264',
        use_all_cores=True,
        model_type="quality"
    )