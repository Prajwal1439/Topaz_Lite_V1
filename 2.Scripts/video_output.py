import subprocess
import shutil
import os
import cv2
import datetime
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Find FFmpeg in system PATH
FFMPEG_PATH = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'

# Cache for probe results to avoid repeated calls
_probe_cache = {}


def get_audio_codec_fast(input_path):
    """Optimized audio codec detection with caching and faster probe"""
    if input_path in _probe_cache:
        return _probe_cache[input_path]['audio_codec']

    # Use ffprobe instead of ffmpeg for faster, JSON-structured output
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'a:0', input_path
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get('streams'):
                codec = data['streams'][0].get('codec_name', '')
                audio_codec = 'copy' if codec in ['aac', 'mp3'] else 'aac'
                _probe_cache[input_path] = {'audio_codec': audio_codec}
                return audio_codec
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass

    # Fallback to original method but with timeout
    try:
        probe_cmd = [FFMPEG_PATH, '-hide_banner', '-i', input_path, '-f', 'null', '-']
        result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, text=True, timeout=5)
        audio_codec = 'copy' if any(codec in result.stderr for codec in ['Audio: aac', 'Audio: mp3']) else 'aac'
        _probe_cache[input_path] = {'audio_codec': audio_codec}
        return audio_codec
    except subprocess.TimeoutExpired:
        return 'aac'  # Safe fallback


def get_video_info_fast(video_path):
    """Fast video info extraction using ffprobe JSON output"""
    if video_path in _probe_cache and 'video_info' in _probe_cache[video_path]:
        return _probe_cache[video_path]['video_info']

    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'v:0', video_path
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get('streams'):
                stream = data['streams'][0]
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                if width > 0 and height > 0:
                    video_info = {'width': width, 'height': height}
                    if video_path not in _probe_cache:
                        _probe_cache[video_path] = {}
                    _probe_cache[video_path]['video_info'] = video_info
                    return video_info
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        pass

    return None


def get_frame_dimensions_fast(frame_dir):
    """Optimized frame dimension detection with early exit and parallel checking"""
    frame_dir = Path(frame_dir)

    # Get first few PNG files for redundancy
    frame_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not frame_files:
        raise RuntimeError("No enhanced frames found")

    # Sort and take first 3 files for redundancy
    frame_files = sorted(frame_files)[:3]

    def read_frame_size(frame_file):
        try:
            frame_path = frame_dir / frame_file
            # Use cv2.IMREAD_UNCHANGED for faster loading (no color conversion)
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                return frame.shape[1], frame.shape[0]  # width, height
        except Exception:
            pass
        return None

    # Try first frame, then parallel check others if needed
    result = read_frame_size(frame_files[0])
    if result:
        return result

    # Parallel check remaining frames if first fails
    if len(frame_files) > 1:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(read_frame_size, f) for f in frame_files[1:]]
            for future in futures:
                try:
                    result = future.result(timeout=2)
                    if result:
                        return result
                except Exception:
                    continue

    raise RuntimeError("Failed to read any frame dimensions")


def finalize_output(input_path, output_path, originalFile, resolution='720p', codec='h264'):
    """Optimized final processing with parallel operations and caching"""
    # --- Path Handling ---
    input_path = Path(input_path).resolve()
    originalFile = Path(originalFile).resolve()
    output_dir = originalFile.parent / "TopazLite_Outputs"
    output_dir.mkdir(exist_ok=True)

    # --- Filename Generation ---
    original_name = originalFile.stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext = '.mov' if codec == 'prores' else '.mp4'
    final_output = f"endResult_{original_name}_{resolution}_{codec}_{timestamp}{output_ext}"
    final_output_path = output_dir / final_output

    # --- Parallel Dimension Detection ---
    width = height = None

    def get_frame_dims():
        try:
            enhanced_dir = input_path.parent / "enhanced_frames"
            return get_frame_dimensions_fast(enhanced_dir)
        except Exception as e:
            print(f"[⚠️] Frame dimension detection failed: {e}")
            return None

    def get_video_dims():
        try:
            info = get_video_info_fast(str(originalFile))
            return (info['width'], info['height']) if info else None
        except Exception as e:
            print(f"[⚠️] Video dimension detection failed: {e}")
            return None

    # Try both methods in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_future = executor.submit(get_frame_dims)
        video_future = executor.submit(get_video_dims)

        # Get frame dimensions first (preferred)
        try:
            frame_result = frame_future.result(timeout=10)
            if frame_result:
                width, height = frame_result
                print(f"[ℹ️] Frame dimensions: {width}x{height} (from enhanced frames)")
        except Exception:
            pass

        # Fallback to video dimensions if frame detection failed
        if width is None or height is None:
            try:
                video_result = video_future.result(timeout=5)
                if video_result:
                    width, height = video_result
                    print(f"[ℹ️] Using original dimensions: {width}x{height}")
            except Exception:
                pass

    if width is None or height is None:
        raise RuntimeError("All dimension detection methods failed")

    # --- Your Original Aspect Ratio Logic (Preserved Exactly) ---
    original_ar = width / height
    print(f"[ℹ️] Original aspect ratio: {original_ar:.2f}")

    # Resolution targets
    TARGETS = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160)
    }

    # Smart scaling that maintains original aspect ratio
    if original_ar >= 1:  # Landscape or square
        base_width, base_height = TARGETS.get(resolution, (width, height))
        target_height = base_height
        target_width = int(target_height * original_ar)
    else:  # Portrait
        base_height, base_width = TARGETS.get(resolution, (height, width))
        target_width = base_width
        target_height = int(target_width / original_ar)

    # Ensure even dimensions
    target_width = target_width + (target_width % 2)
    target_height = target_height + (target_height % 2)
    print(f"[ℹ️] Target dimensions: {target_width}x{target_height}")

    # --- Optimized Video Processing ---
    vf = [
        f"scale={target_width}:{target_height}:flags=lanczos",
        "unsharp=5:5:0.8:3:3:0.4"
    ]

    # --- Codec Configuration with Quality Priority ---
    codec_config = {
        'h264': ['-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-x264-params', 'threads=auto'],
        'h265': ['-c:v', 'libx265', '-crf', '20', '-preset', 'slow', '-x265-params', 'pools=+'],
        'prores': ['-c:v', 'prores_ks', '-profile:v', '3']
    }.get(codec)

    if not codec_config:
        raise ValueError(f"Unsupported codec: {codec}")

    # --- Parallel Audio Processing ---
    def get_audio_info():
        return get_audio_codec_fast(str(originalFile))

    with ThreadPoolExecutor(max_workers=1) as executor:
        audio_future = executor.submit(get_audio_info)

        # Continue with other preparations while audio detection runs
        # --- FFmpeg Command Preparation ---
        base_cmd = [
            FFMPEG_PATH, '-y',
            '-threads', '0',  # Use all CPU cores
            '-i', str(input_path),  # Frame-merged video
            '-i', str(originalFile),  # Original for audio
            '-map', '0:v',  # Video from merged frames
            '-map', '1:a',  # Audio from original
            '-vf', ','.join(vf),
            *codec_config,
        ]

        # Get audio codec result
        try:
            audio_codec = audio_future.result(timeout=5)
        except Exception:
            audio_codec = 'aac'  # Safe fallback

    # --- Audio Handling ---
    audio_args = ['-c:a', audio_codec]
    if audio_codec == 'aac':
        audio_args.extend(['-b:a', '192k', '-ac', '2'])  # Limit to stereo for efficiency

    # --- Final Command ---
    cmd = base_cmd + audio_args + [
        '-movflags', '+faststart',
        '-avoid_negative_ts', 'make_zero',  # Prevent timestamp issues
        str(final_output_path)
    ]

    # --- Execute with Optimizations ---
    try:
        print(f"[⚙️] Running optimized FFmpeg command...")
        # Use lower-level subprocess options for better performance
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered for real-time output
            universal_newlines=True
        )

        # Monitor progress without blocking
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

        # --- Fast Validation ---
        output_path_obj = Path(final_output_path)
        if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
            raise RuntimeError("Output file was not created or is empty")

        print(f"[✅] Successfully created: {final_output_path}")
        return str(final_output_path)

    except subprocess.CalledProcessError as e:
        error_msg = [
            "FFmpeg processing failed!",
            f"Exit code: {e.returncode}",
            "Error output (last 500 chars):",
            (e.stderr or "No error output")[-500:]
        ]
        raise RuntimeError('\n'.join(error_msg))
    except Exception as e:
        raise RuntimeError(f"Unexpected error during processing: {str(e)}")


def clear_cache():
    """Clear the probe cache if memory usage becomes a concern"""
    global _probe_cache
    _probe_cache.clear()


if __name__ == "__main__":
    # Example usage (adjust paths as needed)
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