import subprocess
import shutil
import os
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
    """Finalize video output with enhanced naming and organization while preserving original functionality.

    Args:
        input_path: Path to intermediate video file
        output_path: Desired output path (without extension)
        originalFile: Path to original source video
        resolution: Output resolution ('720p', '1080p', '4k')
        codec: Output codec ('h264', 'h265', 'prores')

    Returns:
        str: Path to the finalized video file

    Raises:
        ValueError: For invalid resolution/codec
        RuntimeError: If FFmpeg processing fails
    """
    # Convert paths to Linux format
    input_path = input_path.replace('\\', '/')
    originalFile = originalFile.replace('\\', '/')

    # Create organized output directory
    output_dir = os.path.join(os.path.dirname(originalFile), "TopazLite_Outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Generate distinctive output filename
    original_name = os.path.splitext(os.path.basename(originalFile))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext = '.mov' if codec == 'prores' else '.mp4'
    final_output = f"endResult_{original_name}_{resolution}_{codec}_{timestamp}{output_ext}"
    output_path = os.path.join(output_dir, final_output)

    # Video scaling and filters
    scale_filter = {
        '720p': "scale='trunc(oh*a/2)*2:720':flags=lanczos",
        '1080p': "scale='trunc(oh*a/2)*2:1080':flags=lanczos",
        '4k': "scale='trunc(oh*a/2)*2:2160':flags=lanczos"
    }
    unsharp_filter = "unsharp=5:5:0.8:3:3:0.4"

    if resolution not in scale_filter:
        raise ValueError("Unsupported resolution. Choose '720p', '1080p', or '4k'.")

    vf = f"{scale_filter[resolution]},{unsharp_filter}"

    # Codec configuration (preserving original settings)
    if codec == 'h264':
        codec_args = ['-c:v', 'libx264', '-crf', '18']
    elif codec == 'h265':
        codec_args = ['-c:v', 'libx265', '-crf', '20']
    elif codec == 'prores':
        codec_args = ['-c:v', 'prores_ks', '-profile:v', '3']
    else:
        raise ValueError("Unsupported codec. Choose 'h264', 'h265', or 'prores'.")

    # Audio handling
    audio_codec = get_audio_codec(originalFile)

    # Build FFmpeg command (preserving original structure)
    cmd = [
        FFMPEG_PATH, '-y',
        '-i', input_path,
        '-i', originalFile,
        '-map', '0:v',
        '-map', '1:a',
        '-vf', vf,
        *codec_args,
        '-c:a', audio_codec,
        '-movflags', '+faststart',
        output_path
    ]

    # Add audio bitrate if re-encoding
    if audio_codec == 'aac':
        cmd.extend(['-b:a', '192k'])

    # Execute with enhanced error handling
    try:
        print(f"[ℹ️] Finalizing output to: {output_path}")
        subprocess.run(cmd, check=True)

        # Verify output
        if not os.path.exists(output_path):
            raise RuntimeError("Output file was not created")
        if os.path.getsize(output_path) == 0:
            raise RuntimeError("Output file is empty")

        print(f"[✅] Successfully saved: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = [
            "FFmpeg processing failed!",
            f"Command: {' '.join(cmd)}",
            f"Exit code: {e.returncode}",
            "Error output:",
            e.stderr.strip()[:500]  # Show first 500 chars of error
        ]
        raise RuntimeError('\n'.join(error_msg))
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


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