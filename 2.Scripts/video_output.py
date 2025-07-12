import subprocess
import shutil

# Find FFmpeg in system PATH
FFMPEG_PATH = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'


def finalize_output(input_path, output_path, resolution='720p', codec='h264'):
    # Convert paths to Linux format
    input_path = input_path.replace('\\', '/')
    output_path = output_path.replace('\\', '/')

    scale_filter = {
        '720p': "scale=-1:720:flags=lanczos",
        '1080p': "scale=-1:1080:flags=lanczos"
    }
 
    unsharp_filter = "unsharp=5:5:0.8:3:3:0.4"

    if resolution not in scale_filter:
        raise ValueError("Unsupported resolution. Choose '720p' or '1080p'.")

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
        '-c:a', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[❌] FFmpeg command failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        raise e
    except Exception as e:
        print(f"[❌] Error during video processing: {str(e)}")
        raise e

    print(f"[✅] Finalized and saved: {output_path}")