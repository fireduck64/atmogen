import os
import sys
import shutil
import subprocess
import torch
from .config import AtmogenConfig

def compile_video(config: AtmogenConfig, input_dir: str, output_path: str):
    """
    Compiles the sparse frames into a final video using GPU-accelerated RIFE frame interpolation.
    """
    print(f"Compiling video from {input_dir} to {output_path} using RIFE frame interpolation...")
    
    # Check if we have ffmpeg installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in PATH.")
        return

    # Paths to RIFE script and model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rife_script = os.path.join(base_dir, "rife_src", "rife.py")
    rife_model = os.path.join(base_dir, "rife_src", "model", "flownet-v46.pkl")

    if not os.path.exists(rife_script) or not os.path.exists(rife_model):
        print(f"Error: RIFE implementation not found. Expected script at {rife_script} and model at {rife_model}")
        return

    # Calculate interpolation multiplier (target_fps / sparse_fps)
    multi = int(round(config.video.output_fps / config.video.sparse_fps))
    print(f"RIFE configuration: output_fps={config.video.output_fps}, sparse_fps={config.video.sparse_fps} -> multiplier={multi}")

    # Set up temporary directory for interpolated frames
    temp_frames_dir = os.path.join(input_dir, "rife_interpolated")
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
    os.makedirs(temp_frames_dir, exist_ok=True)

    # Step 1: Run RIFE frame interpolation
    rife_cmd = [
        sys.executable, rife_script,
        "--model", rife_model,
        "--input", input_dir,
        "--output", temp_frames_dir,
        "--multi", str(multi),
        "--buffer", "0",
        "--change", "0.0"
    ]
    if torch.cuda.is_available():
        rife_cmd.append("--fp16")

    print("Running RIFE interpolation... This will be extremely fast on GPU.")
    print("Command:", " ".join(rife_cmd))
    
    rife_process = subprocess.Popen(rife_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in rife_process.stdout:
        print(line, end="")
    rife_process.wait()

    if rife_process.returncode != 0:
        print("Error: RIFE frame interpolation failed.")
        return

    # Step 2: Compile interpolated frames to video using FFmpeg
    input_pattern = os.path.join(temp_frames_dir, "%06d.jpg")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y", # overwrite output
        "-framerate", str(config.video.output_fps),
        "-i", input_pattern,
        "-c:v", "libx265",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    print("Running FFmpeg encoding...")
    print("Command:", " ".join(ffmpeg_cmd))
    
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in ffmpeg_process.stdout:
        print(line, end="")
    ffmpeg_process.wait()

    # Step 3: Clean up temporary interpolated frames
    if os.path.exists(temp_frames_dir):
        print(f"Cleaning up temporary frames in {temp_frames_dir}...")
        shutil.rmtree(temp_frames_dir)

    if ffmpeg_process.returncode == 0:
        print(f"Video compiled successfully: {output_path}")
    else:
        print("Error compiling video.")
