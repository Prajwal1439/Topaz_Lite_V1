import subprocess
import shutil

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
    # Convert paths to Linux format
    input_path = input_path.replace('\\', '/')
    output_path = output_path.replace('\\', '/')

    scale_filter = {
        '720p': "scale='trunc(oh*a/2)*2:720':flags=lanczos",
        '1080p': "scale='trunc(oh*a/2)*2:1080':flags=lanczos",
        '4k': "scale='trunc(oh*a/2)*2:2160':flags=lanczos"
    }
    unsharp_filter = "unsharp=5:5:0.8:3:3:0.4"

    if resolution not in scale_filter:
        raise ValueError("Unsupported resolution. Choose '720p', '1080p', or '4k'.")

    vf = f"{scale_filter[resolution]},{unsharp_filter}"

    if codec == 'h264':
        codec_args = ['-c:v', 'libx264', '-crf', '18']
    elif codec == 'h265':
        codec_args = ['-c:v', 'libx265', '-crf', '20']
    elif codec == 'prores':
        codec_args = ['-c:v', 'prores_ks', '-profile:v', '3']
    else:
        raise ValueError("Unsupported codec. Choose 'h264', 'h265', or 'prores'.")

    audio_codec = get_audio_codec(originalFile)

    # Build base command
    cmd = [
        FFMPEG_PATH, '-y',
        '-i', input_path,
        '-i', originalFile,
        '-map', '0:v',  # Video from first input
        '-map', '1:a',  # Audio from original file
        '-vf', vf,
        *codec_args,
        '-c:a', audio_codec,
        '-movflags', '+faststart',
        output_path
    ]

    # Add audio bitrate only if we're re-encoding
    if audio_codec == 'aac':
        cmd.extend(['-b:a', '192k'])

    try:
        subprocess.run(cmd, check=True)
        print(f"[✅] Finalized and saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[❌] FFmpeg command failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        raise e
    except Exception as e:
        print(f"[❌] Error during video processing: {str(e)}")
        raise e


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