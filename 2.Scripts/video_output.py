import subprocess
import shutil
import os
import cv2
import datetime
import subprocess
from pathlib import Path

# Find FFmpeg in system PATH
FFMPEG_PATH = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'


def get_audio_codec(input_path):
    """Detect if audio needs re-encoding (returns 'copy' or 'aac')"""
    probe_cmd = [
        FFMPEG_PATH, '-hide_banner', '-i', input_path
    ]
    result = subprocess.run(
        probe_cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    # Check for AAC/MP3 (safe to copy)
    if 'Audio: aac' in result.stderr or 'Audio: mp3' in result.stderr:
        return 'copy'
    return 'aac'  # Re-encode if other codec (AC3, FLAC, etc.)


def finalize_output(input_path, output_path, originalFile, resolution='720p', codec='h264'):
    """Final processing for frame-merged videos without metadata"""
    # --- Path Handling ---
    input_path = str(Path(input_path).resolve())
    originalFile = str(Path(originalFile).resolve())
    output_dir = Path(originalFile).parent / "TopazLite_Outputs"
    output_dir.mkdir(exist_ok=True)

    # --- Filename Generation ---
    original_name = Path(originalFile).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext = '.mov' if codec == 'prores' else '.mp4'
    final_output = f"endResult_{original_name}_{resolution}_{codec}_{timestamp}{output_ext}"
    output_path = str(output_dir / final_output)

    # --- Dimension Handling for Frame-Merged Videos ---
    def get_frame_dimensions(frame_dir):
        """Get dimensions from first frame in enhanced_frames directory"""
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        if not frame_files:
            raise RuntimeError("No enhanced frames found")

        first_frame = cv2.imread(str(Path(frame_dir) / frame_files[0]))
        if first_frame is None:
            raise RuntimeError("Failed to read first frame")
        return first_frame.shape[1], first_frame.shape[0]  # width, height

    try:
        # Get dimensions from frames rather than video
        enhanced_dir = str(Path(input_path).parent / "enhanced_frames")
        width, height = get_frame_dimensions(enhanced_dir)
        print(f"[ℹ️] Frame dimensions: {width}x{height} (from enhanced frames)")
    except Exception as e:
        print(f"[⚠️] Could not get frame dimensions: {str(e)}")
        try:
            # Ultimate fallback: use original video dimensions
            probe_cmd = [FFMPEG_PATH, '-hide_banner', '-i', originalFile]
            result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, text=True)
            for line in result.stderr.split('\n'):
                if 'Video:' in line and 'x' in line:
                    parts = [p.strip() for p in line.split(',')]
                    dimension_part = next(p for p in parts if 'x' in p and 'tbr' not in p)
                    width, height = map(int, dimension_part.split()[0].split('x'))
                    print(f"[ℹ️] Using original dimensions: {width}x{height}")
                    break
        except Exception as fallback_e:
            raise RuntimeError(f"All dimension detection failed:\n{str(e)}\nFallback also failed:\n{str(fallback_e)}")

    # --- Aspect Ratio Preservation ---
    original_ar = width / height
    print(f"[ℹ️] Original aspect ratio: {original_ar:.2f}")

    # --- Resolution Targets ---
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

    # --- Video Processing ---
    vf = [
        f"scale={target_width}:{target_height}:flags=lanczos",
        "unsharp=5:5:0.8:3:3:0.4"
    ]

    # --- Codec Configuration ---
    codec_config = {
        'h264': ['-c:v', 'libx264', '-crf', '18', '-preset', 'slow'],
        'h265': ['-c:v', 'libx265', '-crf', '20', '-preset', 'slow'],
        'prores': ['-c:v', 'prores_ks', '-profile:v', '3']
    }.get(codec)

    if not codec_config:
        raise ValueError(f"Unsupported codec: {codec}")

    # --- Audio Handling ---
    audio_codec = get_audio_codec(originalFile)
    audio_args = ['-c:a', audio_codec]
    if audio_codec == 'aac':
        audio_args.extend(['-b:a', '192k'])

    # --- FFmpeg Command ---
    cmd = [
        FFMPEG_PATH, '-y',
        '-i', input_path,  # Frame-merged video
        '-i', originalFile,  # Original for audio
        '-map', '0:v',  # Video from merged frames
        '-map', '1:a',  # Audio from original
        '-vf', ','.join(vf),
        *codec_config,
        *audio_args,
        '-movflags', '+faststart',
        output_path
    ]

    # --- Execute with Validation ---
    try:
        print(f"[⚙️] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Verify output
        if not Path(output_path).exists():
            raise RuntimeError("Output file was not created")
        if Path(output_path).stat().st_size == 0:
            raise RuntimeError("Output file is empty")

        print(f"[✅] Successfully created: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = [
            "FFmpeg processing failed!",
            f"Command: {' '.join(cmd)}",
            f"Exit code: {e.returncode}",
            "Error output:",
            e.stderr.strip()[:1000]  # Show first 1000 chars
        ]
        raise RuntimeError('\n'.join(error_msg))


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