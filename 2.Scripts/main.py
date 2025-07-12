import os
import subprocess
import cv2
import multiprocessing
import gc
import psutil
import torch
from functools import partial
from tqdm import tqdm
from upscale_frame import upscale_image
from video_output import finalize_output

# Global variables for shared resources
global_upsampler = None


def init_worker():
    """Initialize worker process with proper resource limits"""
    import torch
    torch.set_num_threads(1)  # Limit threads per worker
    gc.collect()


def clean_memory():
    """Force clean memory and cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_system_resources(use_all_cores=True):
    """
    Calculate optimal number of workers based on system resources
    Args:
        use_all_cores (bool): Whether to use all available cores
    Returns:
        int: Number of recommended worker processes
    """
    total_cores = multiprocessing.cpu_count()
    available_mem = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB

    # Memory requirement per worker (adjust based on your needs)
    mem_per_worker = 2.5  # GB

    # Calculate maximum safe workers based on memory
    mem_based_workers = int(available_mem / mem_per_worker)

    # Core-based calculation
    if use_all_cores:
        core_based_workers = total_cores
    else:
        core_based_workers = max(1, total_cores - 1)  # Leave one core free

    # Use the more restrictive limit
    safe_workers = min(core_based_workers, mem_based_workers)

    # Ensure at least 1 worker
    return max(1, safe_workers)


def extract_frames(video_path, frame_dir):
    """Optimized frame extraction with hardware acceleration"""
    os.makedirs(frame_dir, exist_ok=True)
    input_path = os.path.abspath(video_path).replace("\\", "/")
    output_path = os.path.abspath(os.path.join(frame_dir, "frame_%06d.png")).replace("\\", "/")

    # Get video metadata
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


def process_single_frame(file_info, enhanced_dir):
    """Optimized frame processing with memory management"""
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

        with torch.no_grad():
            upscale_image(img, output_path, scaling)

        return True
    except Exception as e:
        print(f"[❌] Error processing {file}: {str(e)}")
        return False
    finally:
        del img
        clean_memory()


def process_frame_wrapper(args):
    """
    Properly structured wrapper for frame processing
    Args:
        args: Tuple of (file_name, input_path, enhanced_dir)
    """
    try:
        file_name, input_path, enhanced_dir = args  # Explicit unpacking
        output_path = os.path.join(enhanced_dir, file_name)
        return process_single_frame((file_name, input_path), enhanced_dir)
    except Exception as e:
        print(f"[⚠️] Error in wrapper for {args[0]}: {str(e)}")
        return False


def enhance_frames(frame_dir, enhanced_dir, use_all_cores=True):
    """Optimized frame enhancement with proper argument handling"""
    # Validate directories
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    # Get sorted frame files
    frame_files = sorted([
        (f, os.path.join(frame_dir, f))
        for f in os.listdir(frame_dir)
        if f.lower().endswith('.png') and os.path.isfile(os.path.join(frame_dir, f))
    ], key=lambda x: int(''.join(filter(str.isdigit, x[0]))))

    if not frame_files:
        raise ValueError(f"No valid PNG frames found in: {frame_dir}")

    # Prepare output directory
    os.makedirs(enhanced_dir, exist_ok=True)

    # Calculate workers (simplified)
    num_workers = min(multiprocessing.cpu_count(), len(frame_files))
    if not use_all_cores:
        num_workers = max(1, num_workers - 1)

    print(f"[ℹ️] Processing {len(frame_files)} frames with {num_workers} workers")

    # Prepare arguments
    args = [
        (file_name, file_path, os.path.abspath(enhanced_dir))
        for file_name, file_path in frame_files
    ]

    try:
        with multiprocessing.Pool(
                processes=num_workers,
                initializer=init_worker,
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

            # Verify output
            output_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.png')]
            if len(output_files) != success_count:
                print(f"[⚠️] Output mismatch: {len(output_files)} files vs {success_count} successes")

    except Exception as e:
        print(f"[❌] Frame processing failed: {str(e)}")
        raise


def assemble_video(enhanced_dir, output_path, fps):
    """Assemble enhanced frames into a video with thorough error checking"""
    # Verify enhanced_dir exists and contains frames
    if not os.path.exists(enhanced_dir):
        raise FileNotFoundError(f"Enhanced frames directory not found: {enhanced_dir}")

    frame_files = sorted([f for f in os.listdir(enhanced_dir) if f.endswith('.png')])
    if not frame_files:
        raise ValueError(f"No PNG frames found in: {enhanced_dir}")

    # Verify the first frame exists
    first_frame = os.path.join(enhanced_dir, frame_files[0])
    if not os.path.exists(first_frame):
        raise FileNotFoundError(f"First frame not found: {first_frame}")

    # Prepare FFmpeg command
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(enhanced_dir, 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path
    ]

    print(f"[🔁] Assembling {len(frame_files)} frames into video at {fps} FPS")
    print(f"Output will be saved to: {output_path}")

    try:
        # Run FFmpeg with timeout (60 seconds)
        subprocess.run(
            cmd,
            check=True,
            timeout=60,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(f"[✅] Successfully assembled video: {output_path}")

        # Verify output file was created
        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg completed but output file was not created")
        if os.path.getsize(output_path) == 0:
            raise RuntimeError("Output video file is empty")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Video assembly timed out after 60 seconds")
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed with code {e.returncode}\n"
        error_msg += f"Command: {' '.join(cmd)}\n"
        error_msg += f"Error output:\n{e.stderr}"
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Video assembly failed: {str(e)}")


def process_video(video_path, temp_dir='temp', resolution='720p', codec='h264', use_all_cores=True):
    """Main video processing pipeline with cleanup"""
    try:
         # Setup directories
        frame_dir = os.path.join(temp_dir, 'frames')
        enhanced_dir = os.path.join(temp_dir, 'enhanced_frames')
        intermediate_video = os.path.join(temp_dir, 'enhanced_output.mp4')
        final_output = os.path.splitext(video_path)[0] + f'_final_{resolution}.mp4'

        # Processing pipeline
        fps = extract_frames(video_path, frame_dir)
        enhance_frames(frame_dir, enhanced_dir, use_all_cores)
        assemble_video(enhanced_dir, intermediate_video, fps)
        finalize_output(intermediate_video, final_output, resolution, codec)

        print(f"[✅] Final video saved as: {final_output}")
    finally:
        clean_memory()


if __name__ == '__main__':
    process_video(
        video_path='3.Test_Images_Or_Videos/sample_input.mp4',
        temp_dir='temp_processing',
        resolution='720p',
        codec='h264',
        use_all_cores=True
    )