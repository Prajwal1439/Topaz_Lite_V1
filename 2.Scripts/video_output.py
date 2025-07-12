import subprocess
import shutil

# Find FFmpeg in system PATH
FFMPEG_PATH = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'


def finalize_output(input_path, output_path, resolution='720p', codec='h264'):
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

    cmd = [
        FFMPEG_PATH, '-y', '-i', input_path,
        '-vf', vf,
        *codec_args,
        '-c:a', 'aac', '-b:a', '192k',  # Added proper audio encoding
        '-movflags', '+faststart',       # Better for web streaming
        output_path
    ]

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
    input_video = "3.Test_Images_Or_Videos/Sample.mp4"  # Input video (not frames!)
    output_video = "4.Outputs/final_output.mp4"         # Output path

    finalize_output(
        input_path=input_video,
        output_path=output_video,
        resolution="720p",  # or "1080p"
        codec="h264"        # or "h265"/"prores"
    )