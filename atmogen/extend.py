import os
import sys
import shutil
import subprocess
import torch
import numpy as np
from PIL import Image
from .config import AtmogenConfig
from .library import get_asset_path


def extract_frames_to_sparse_timeline(input_video: str, output_dir: str, config: AtmogenConfig) -> int:
    """Extract all frames from the input video at native framerate for RIFE interpolation."""
    os.makedirs(output_dir, exist_ok=True)

    # Get duration of input video
    probe_duration_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    result = subprocess.run(probe_duration_cmd, capture_output=True, text=True)
    input_duration = float(result.stdout.strip())

    # Get source video's frame rate via ffprobe (video stream side of the probe)
    probe_fps_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    result_fps = subprocess.run(probe_fps_cmd, capture_output=True, text=True)
    fps_str = result_fps.stdout.strip()
    
    # Parse frame rate (can be a/b format like "24000/1001" or integer like "24")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        src_fps = float(num) / float(den)
    else:
        src_fps = float(fps_str)
        if src_fps == 0:
            src_fps = 24.0

    print(f"Input video duration: {input_duration}s, source fps: {src_fps:.2f}")

    # Extract ALL frames at native framerate using the -r flag in the output
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", f"fps={src_fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.png")
    ]

    extract_process = subprocess.Popen(
        extract_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = extract_process.communicate()

    if extract_process.returncode != 0:
        print(f"Error extracting frames: {stderr}")
        return 0

    actual_frames = len([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".png")])
    if actual_frames == 0:
        print("Warning: No frames were extracted. Check the input video.")
        return 0

    print(f"Extracted {actual_frames} frames from input video (named frame_*.png for R compatibility).")
    return actual_frames


def upscale_to_target(input_dir: str, config: AtmogenConfig):
    """Upscale all PNG frames in the directory to the config target resolution using FFmpeg."""
    print(f"Upscaling all frames to {config.video.width}x{config.video.height}...")

    frame_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
    ])

    if not frame_files:
        print("Error: No frames found to upscale.")
        return

    target_w = config.video.width
    target_h = config.video.height

    for frame_name in frame_files:
        input_frame = os.path.join(input_dir, frame_name)
        upscale_cmd = [
            "ffmpeg",
            "-y",
            "-i", input_frame,
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos,format=rgb24",
            input_frame
        ]
        result = subprocess.run(upscale_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            print(f"Error scaling {frame_name}: {result.stderr}")

    print("Upscaling complete.")


def compile_extended_video(config: AtmogenConfig, input_dir: str, output_path: str):
    """
    Extend the video using RIFE frame interpolation to reach the target duration.
    The extracted sparse frames are the input; RIFE multiplies them to fill the
    target duration at output_fps.
    """
    print(f"Extending to target duration ({config.video.duration_minutes} min, {config.video.output_fps} fps) using RIFE...")

    # Check if ffmpeg is available
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
        print(f"Error: RIFE implementation not found. Script: {rife_script}, Model: {rife_model}")
        return

    # Calculate total target frames
    target_frame_count = int(config.video.duration_minutes * 60 * config.video.output_fps)

    # Count sparse source frames
    source_frames = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
    ])

    num_source = len(source_frames)
    if num_source < 2:
        print("Error: Need at least 2 source frames for RIFE interpolation.")
        return

    # Calculate RIFE multiplier
    # With --multi N, each gap produces (N-1) interpolated frames
    # Total output = num_source + (num_source - 1) * (multi - 1)
    # Solving for multi: multi = (target - num_source) / (num_source - 1) + 1
    multi = max(2, int(np.ceil((target_frame_count - num_source) / (num_source - 1))) + 1)

    print(f"Source sparse frames: {num_source}")
    print(f"Target total frames: {target_frame_count}")
    print(f"RIFE multiplier: {multi} (per gap between frames)")

    # Set up output dir for RIFE
    interpolated_dir = os.path.join(input_dir, "rife_interpolated")
    if os.path.exists(interpolated_dir):
        shutil.rmtree(interpolated_dir)
    os.makedirs(interpolated_dir, exist_ok=True)

    rife_cmd = [
        sys.executable, rife_script,
        "--model", rife_model,
        "--input", input_dir,
        "--output", interpolated_dir,
        "--multi", str(multi),
        "--buffer", "0",
        "--change", "0.0"
    ]
    if torch.cuda.is_available():
        rife_cmd.append("--fp16")

    print("Running RIFE interpolation...")
    rife_proc = subprocess.Popen(
        rife_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in rife_proc.stdout:
        print(line, end="")
    rife_proc.wait()

    if rife_proc.returncode != 0:
        print("RIFE interpolation failed.")
        return

    # Encode extended frames to silent video at config FPS
    frame_pattern = os.path.join(interpolated_dir, "%06d.jpg")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(config.video.output_fps),
        "-i", frame_pattern,
        "-c:v", "libx265",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",  # No audio
        output_path
    ]

    print("Encoding extended silent video with FFmpeg...")
    enc_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in enc_proc.stdout:
        print(line, end="")
    enc_proc.wait()

    # Clean up
    if os.path.exists(interpolated_dir):
        shutil.rmtree(interpolated_dir)

    if enc_proc.returncode == 0:
        print(f"Extended video compiled: {output_path}")
    else:
        print("Encoding failed.")


def mux_soundtrack(config: AtmogenConfig, video_path: str, output_dir: str, final_output: str):
    """Mux a resolved audio asset from the library into the extended video."""
    if not os.path.exists(video_path):
        print(f"Error: Extended video not found at {video_path}")
        print("Please check RIFE output and ensure the video was compiled successfully.")
        return

    if not config.soundtrack or not config.soundtrack.enabled:
        print("Soundtrack is disabled. Copying extended video to output.")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-c", "copy", final_output],
                       capture_output=True)
        print(f"Final output: {final_output}")
        return

    # Resolve asset from the library using slug and revision
    asset_path = get_asset_path(config.soundtrack.asset_slug, config.soundtrack.asset_revision)
    if not asset_path:
        print(f"Error: Audio asset '{config.soundtrack.asset_slug}' (v{config.soundtrack.asset_revision}) not found in library.")
        print("Please generate it first using 'python main.py audio-gen <audio_config>.yaml'")
        return

    # Copy from library to output dir for final muxing
    final_audio = os.path.join(output_dir, "soundtrack.mp3")
    shutil.copy2(asset_path, final_audio)

    # Mux extended silent video + soundtrack
    print("Muxing video and soundtrack into final video...")
    mux_cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", final_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_output
    ]
    result = subprocess.run(mux_cmd)
    if result.returncode == 0:
        print(f"Final extended video with audio complete: {final_output}")
    else:
        print("Audio muxing failed.")
