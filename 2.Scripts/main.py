import os
import subprocess
import cv2
import multiprocessing
from functools import partial
from tqdm import tqdm

from upscale_frame import upscale_image
from video_output import finalize_output


def init_worker():
    """Initialize worker process"""
    import torch
    torch.set_num_threads(1)  # Limit threads per worker
    gc.collect()

def clean_memory():
    """Force clean memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)

    # Linux path handling
    input_path = os.path.abspath(video_path).replace("\\", "/")
    output_path = os.path.abspath(os.path.join(frame_dir, "frame_%06d.png")).replace("\\", "/")

    # First get the video's FPS
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30  # Fallback to 30 FPS if we can't determine it
        print(f"[⚠️] Couldn't determine FPS, defaulting to {fps}")

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        output_path
    ]

    print(f"[🔁] Extracting frames at {fps} FPS:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return fps  # Return the detected FPS
    except subprocess.CalledProcessError as e:
        print(f"[❌] FFmpeg failed with code {e.returncode}")
        raise e
    except FileNotFoundError:
        print("[❌] FFmpeg not found. Install with: sudo apt install ffmpeg")
        raise


def process_single_frame(file_info, enhanced_dir):
    """Process a single frame with all required arguments in one tuple"""
    file, input_path = file_info
    output_path = os.path.abspath(os.path.join(enhanced_dir, file))

    try:
        if not os.path.isfile(input_path):
            print(f"[⚠️] File disappeared: {input_path}")
            return False

        img = cv2.imread(input_path.replace('\\', '/'), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[⚠️] Failed to decode image: {input_path}")
            return False

        h, w = img.shape[:2]
        min_dim = min(h, w)

        if min_dim >= 1080:
            scale = 720 / min_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            upscale_image(resized, output_path, 1.5)
        elif min_dim == 720:
            upscale_image(input_path, output_path, 1.5)
        else:
            upscale_image(input_path, output_path)

        return True
    except Exception as e:
        print(f"[❌] Error processing {file}: {str(e)}")
        return False


def enhance_frames(frame_dir, enhanced_dir, use_all_cores=True):
    os.makedirs(enhanced_dir, exist_ok=True)

    # Get and validate PNG files
    frame_files = []
    for f in os.listdir(frame_dir):
        if f.lower().endswith('.png'):
            full_path = os.path.abspath(os.path.join(frame_dir, f))
            if os.path.isfile(full_path):
                frame_files.append((f, full_path))

    # Sort frames numerically
    def frame_key(x):
        try:
            return int(''.join(filter(str.isdigit, x[0])))
        except ValueError:
            return 0

    frame_files.sort(key=frame_key)

    # Determine number of cores to use
    total_cores = multiprocessing.cpu_count()
    if use_all_cores:
        num_workers = total_cores
    else:
        num_workers = max(1, total_cores - 1)  # Always leave at least 1 core free

    print(f"[ℹ️] Using {num_workers} out of {total_cores} available CPU cores")

    # Process frames in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Create a list of arguments for each frame
        args = [(file_info, enhanced_dir) for file_info in frame_files]

        # Use tqdm for progress bar
        results = list(tqdm(
            pool.starmap(process_single_frame, args),
            total=len(frame_files),
            desc="[🔁] Processing frames"
        ))

    success_count = sum(results)
    print(f"[✅] Processed {success_count}/{len(frame_files)} frames successfully")


def assemble_video(enhanced_dir, output_path, fps):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(enhanced_dir, 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path
    ]
    print(f"[🔁] Assembling video at {fps} FPS")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        print(f"[❌] Failed to run command: {cmd}")
        raise e


def process_video(video_path, temp_dir='temp', resolution='720p', codec='h264', use_all_cores=True):
    frame_dir = os.path.join(temp_dir, 'frames')
    enhanced_dir = os.path.join(temp_dir, 'enhanced_frames')
    intermediate_video = os.path.join(temp_dir, 'enhanced_output.mp4')
    final_output = os.path.splitext(video_path)[0] + f'_final_{resolution}.mp4'

    # Extract frames and get the original FPS
    fps = extract_frames(video_path, frame_dir)

    enhance_frames(frame_dir, enhanced_dir, use_all_cores)
    assemble_video(enhanced_dir, intermediate_video, fps)  # Pass the FPS
    finalize_output(intermediate_video, final_output, resolution, codec)

    print(f"[✅] Final video saved as: {final_output}")


# Run
if __name__ == '__main__':
    process_video(
        video_path='3.Test_Images_Or_Videos/sample_input.mp4',
        temp_dir='temp_processing',
        resolution='720p',  # or '1080p'
        codec='h264',  # or 'h265', 'prores'
        use_all_cores=True  # or False to leave one core free
    )