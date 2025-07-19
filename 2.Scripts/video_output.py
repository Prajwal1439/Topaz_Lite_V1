import subprocess
import shutil
import os
import threading

import cv2
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import functools

# Cached FFmpeg path with thread-safe initialization
_FFMPEG_PATH = None
_FFMPEG_LOCK = threading.Lock()


def get_ffmpeg_path():
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        with _FFMPEG_LOCK:
            if _FFMPEG_PATH is None:  # Double-check locking
                _FFMPEG_PATH = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
                # Verify FFmpeg is executable
                if not os.access(_FFMPEG_PATH, os.X_OK):
                    raise RuntimeError(f"FFmpeg not executable at {_FFMPEG_PATH}")
    return _FFMPEG_PATH


@functools.lru_cache(maxsize=32)
def get_audio_codec(input_path):
    """Ultra-optimized audio codec detection with caching and timeout"""
    probe_cmd = [
        get_ffmpeg_path(),
        '-hide_banner',
        '-loglevel', 'error',
        '-i', input_path,
        '-t', '0',  # Only analyze headers
        '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(
            probe_cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=3  # Aggressive timeout
        )
        # Fast pattern matching
        stderr = result.stderr
        if 'aac' in stderr or 'mp3' in stderr:
            return 'copy'
        return 'aac'
    except subprocess.TimeoutExpired:
        return 'aac'


def parallel_frame_reader(frame_path):
    """Threaded frame reading for dimension detection"""
    img = cv2.imread(frame_path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is not None:
        return img.shape[1], img.shape[0]
    return None


def get_frame_dimensions(frame_dir):
    """Parallel dimension detection with multiple fallbacks"""
    frame_files = sorted([
                             str(Path(frame_dir) / f)
                             for f in os.listdir(frame_dir)
                             if f.endswith('.png')
                         ][:10])  # Only check first 10 frames max

    if not frame_files:
        raise RuntimeError("No enhanced frames found")

    # Parallel reading attempt
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(parallel_frame_reader, frame_files))

    # Get first successful result
    dimensions = next((d for d in results if d is not None), None)
    if dimensions:
        return dimensions

    # Fallback to sequential if parallel fails
    for frame_path in frame_files:
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if img is not None:
            return img.shape[1], img.shape[0]

    raise RuntimeError("All frame reading attempts failed")


def probe_video_dimensions(video_path):
    """Lightning-fast video dimension probing"""
    ffprobe_path = get_ffmpeg_path().replace('ffmpeg', 'ffprobe')
    cmd = [
        ffprobe_path if os.path.exists(ffprobe_path) else get_ffmpeg_path(),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            text=True,
            timeout=5,
            check=True
        )
        return tuple(map(int, result.stdout.strip().split(',')))
    except Exception as e:
        raise RuntimeError(f"Video probing failed: {str(e)}")


def calculate_target_dimensions(width, height, resolution):
    """Precise dimension calculation with aspect ratio preservation"""
    TARGETS = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160)
    }

    original_ar = width / height
    base_width, base_height = TARGETS.get(resolution, (width, height))

    if original_ar >= 1:  # Landscape or square
        target_height = base_height
        target_width = int(target_height * original_ar)
    else:  # Portrait
        target_width = base_width
        target_height = int(target_width / original_ar)

    # Ensure even dimensions using bitwise operation for speed
    target_width = target_width & ~1
    target_height = target_height & ~1

    return target_width, target_height


def build_ffmpeg_command(input_path, original_path, output_path,
                         target_width, target_height, codec):
    """Highly optimized FFmpeg command builder"""
    # Video filters
    vf = [
        f"scale={target_width}:{target_height}:flags=lanczos",
        "unsharp=5:5:0.8:3:3:0.4"
    ]

    # Thread-optimized codec configurations
    cpu_count = os.cpu_count() or 4
    codec_configs = {
        'h264': [
            '-c:v', 'libx264', '-crf', '18', '-preset', 'faster',
            '-x264-params', f'ref=4:me=hex:subme=6:threads={cpu_count}'
        ],
        'h265': [
            '-c:v', 'libx265', '-crf', '20', '-preset', 'faster',
            '-x265-params', f'ref=4:me=hex:subme=4:pools={cpu_count}'
        ],
        'prores': ['-c:v', 'prores_ks', '-profile:v', '3']
    }

    codec_config = codec_configs.get(codec)
    if not codec_config:
        raise ValueError(f"Unsupported codec: {codec}")

    # Audio handling with caching
    audio_codec = get_audio_codec(original_path)
    audio_args = ['-c:a', audio_codec]
    if audio_codec == 'aac':
        audio_args.extend(['-b:a', '192k'])

    return [
        get_ffmpeg_path(), '-y',
        '-hide_banner',
        '-loglevel', 'warning',
        '-i', input_path,
        '-i', original_path,
        '-map', '0:v',
        '-map', '1:a',
        '-vf', ','.join(vf),
        *codec_config,
        *audio_args,
        '-movflags', '+faststart',
        '-threads', str(cpu_count),
        output_path
    ]


def finalize_output(input_path, output_path, originalFile, resolution='720p', codec='h264'):
    """Ultra-optimized video processing pipeline"""
    # --- Path Handling ---
    input_path = str(Path(input_path).resolve())
    original_path = str(Path(originalFile).resolve())
    output_dir = Path(originalFile).parent / "TopazLite_Outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Filename Generation ---
    original_name = Path(originalFile).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext = '.mov' if codec == 'prores' else '.mp4'
    final_output = f"endResult_{original_name}_{resolution}_{codec}_{timestamp}{output_ext}"
    output_path = str(output_dir / final_output)

    # --- Dimension Detection ---
    try:
        # First try enhanced frames
        enhanced_dir = str(Path(input_path).parent / "enhanced_frames")
        width, height = get_frame_dimensions(enhanced_dir)
        print(f"[ℹ️] Frame dimensions: {width}x{height} (parallel read)")
    except Exception as e:
        print(f"[⚠️] Frame detection failed: {str(e)}")
        try:
            # Fallback to video probing
            width, height = probe_video_dimensions(original_path)
            print(f"[ℹ️] Using probed dimensions: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"All dimension detection failed: {str(e)}")

    # --- Target Calculation ---
    target_width, target_height = calculate_target_dimensions(width, height, resolution)
    print(f"[ℹ️] Target dimensions: {target_width}x{target_height}")

    # --- FFmpeg Execution ---
    cmd = build_ffmpeg_command(
        input_path, original_path, output_path,
        target_width, target_height, codec
    )

    try:
        print(f"[⚙️] Running optimized FFmpeg command")
        start_time = datetime.datetime.now()

        # Use Popen with piped output for better control
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
        ) as proc:
            # Process output in real-time
            for line in proc.stderr:
                if 'frame=' in line:
                    print(f"\r[⏱️] Encoding: {line.strip()}", end='')

            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, proc.stdout, proc.stderr
                )

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n[⏱️] Encoding completed in {elapsed:.2f} seconds")

        # Fast output validation
        if not os.path.exists(output_path):
            raise RuntimeError("Output file was not created")
        if os.path.getsize(output_path) == 0:
            raise RuntimeError("Output file is empty")

        print(f"[✅] Successfully created: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = [
            "FFmpeg processing failed!",
            f"Exit code: {e.returncode}",
            "Last 10 error lines:",
            *e.stderr.splitlines()[-10:]
        ]
        raise RuntimeError('\n'.join(error_msg))
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    input_video = "temp_gui_output/enhanced_output.mp4"
    output_video = "4.Outputs/final_output2.mp4"
    original_video = "3.Test_Images_Or_Videos/WhatsApp Video 2025-07-12 at 9.45.27 PM.mp4"

    finalize_output(
        input_path=input_video,
        output_path=output_video,
        originalFile=original_video,
        resolution="720p",
        codec="h264"
    )