import click
from .config import load_config
from .generator import VideoGenerator
from .video import compile_video
from .audio import generate_soundtrack
import os
import subprocess

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

if __name__ == '__main__':
    cli()
