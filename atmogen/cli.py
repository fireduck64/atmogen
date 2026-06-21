import click
from .config import load_config
from .generator import VideoGenerator
from .video import compile_video
from .audio import generate_soundtrack
from .extend import extract_frames_to_sparse_timeline, upscale_to_target, compile_extended_video, mux_soundtrack
import os
import subprocess
import shutil

@click.group()
def cli():
    """Atmogen: A tool for generating long, slow-morphing atmospheric background videos."""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', default=None, help='Directory to save preview images')
def preview(config_file, output_dir):
    """Generate preview keyframes to review the prompts and seeds."""
    config = load_config(config_file)
    print(f"Loaded config: {config.video.duration_minutes} mins video")
    
    final_output_dir = output_dir if output_dir is not None else config.output_dir
    
    generator = VideoGenerator()
    generator.generate_preview(config, final_output_dir)
    print("Preview complete. Check the output directory.")

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-audio', default=None, help='Path to save soundtrack MP3')
def soundtrack(config_file, output_audio):
    """Generate the ambient AI soundtrack for a configuration."""
    config = load_config(config_file)
    final_output_audio = output_audio if output_audio is not None else os.path.join(config.output_dir, "soundtrack.mp3")
    generate_soundtrack(config, final_output_audio)

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', default=None, help='Directory to save sparse frames')
@click.option('--output-video', default=None, help='Path to final video output')
def render(config_file, output_dir, output_video):
    """Generate the full video by interpolating between keyframes."""
    config = load_config(config_file)
    
    final_output_dir = output_dir if output_dir is not None else config.output_dir
    final_output_video = output_video if output_video is not None else os.path.join(final_output_dir, "final_video.mp4")
    
    # Step 1: Generate sparse frames
    generator = VideoGenerator()
    generator.generate_render(config, final_output_dir)
    
    # Step 2: Handle soundtrack and compile video
    os.makedirs(os.path.dirname(final_output_video), exist_ok=True)
    
    if config.soundtrack and config.soundtrack.enabled:
        final_output_audio = os.path.join(final_output_dir, "soundtrack.mp3")
        if not os.path.exists(final_output_audio):
            print("Soundtrack not found. Generating AI soundtrack...")
            generate_soundtrack(config, final_output_audio)
            
        silent_video = final_output_video.replace(".mp4", "_silent.mp4")
        compile_video(config, final_output_dir, silent_video)
        
        print("Muxing video and soundtrack into final video...")
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i", silent_video,
            "-stream_loop", "-1",  # Loops the input audio stream infinitely
            "-i", final_output_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",          # Cuts the stream exactly where the video ends
            final_output_video
        ]
        subprocess.run(mux_cmd)
        
        if os.path.exists(silent_video):
            os.remove(silent_video)
    else:
        compile_video(config, final_output_dir, final_output_video)

@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', default=None, help='Directory to save intermediate and final output')
@click.option('--output-video', default=None, help='Path to final extended video output')
def extend(input_video, config_file, output_dir, output_video):
    """Extend an input video to the config's target duration using RIFE interpolation and a fresh soundtrack."""
    config = load_config(config_file)
    
    final_output_dir = output_dir if output_dir is not None else config.output_dir
    final_output_video = output_video if output_video is not None else os.path.join(final_output_dir, "extended_video.mp4")
    
    # Step 1: Extract sparse frames from input video
    sparse_dir = os.path.join(final_output_dir, "sparse")
    num_extracted = extract_frames_to_sparse_timeline(input_video, sparse_dir, config)
    if num_extracted == 0:
        print("Failed to extract frames from input video.")
        return
    
    # Step 2: Upscale frames to target resolution
    upscale_to_target(sparse_dir, config)
    
    # Step 3: Extend video using RIFE to reach target duration
    extended_video = os.path.join(final_output_dir, "extended_silent.mp4")
    compile_extended_video(config, sparse_dir, extended_video)
    
    # Step 4: Generate soundtrack and mux final video
    os.makedirs(os.path.dirname(final_output_video), exist_ok=True)
    mux_soundtrack(config, extended_video, final_output_dir, final_output_video)
    
    # Cleanup sparse dir
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    
    # Remove silent intermediate file
    silent_path = extended_video
    if os.path.exists(silent_path):
        os.remove(silent_path)
    
    print("Extension complete!")

if __name__ == '__main__':
    cli()
