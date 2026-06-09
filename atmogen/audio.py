import os
import sys
import wave
import struct
import torch
import numpy as np
from tqdm import tqdm
from .config import AtmogenConfig

def save_wav_file(filename, samples, sample_rate=32000):
    """
    Saves a 1D float array of audio samples as a 16-bit PCM WAV file using built-in wave module.
    """
    # Convert float samples (-1.0 to 1.0) to 16-bit PCM integers
    int_samples = [int(max(-32768, min(32767, x * 32767))) for x in samples]
    
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)  # Mono
        f.setsampwidth(2)  # 16-bit PCM (2 bytes)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(int_samples)}h", *int_samples))

def generate_soundtrack(config: AtmogenConfig, output_path: str):
    """
    Generates a continuous, high-fidelity ambient soundtrack using either local GPU MusicGen or ElevenLabs Music v1.
    """
    # Check if the soundtrack already exists on disk to save API costs/credits
    if os.path.exists(output_path):
        print(f"Soundtrack already exists at {output_path}. Skipping generation to save API costs and credits!")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Formulate dynamic prompt blending visual elements with the soundtrack style
    visual_keywords = []
    for kf in config.keyframes[:3]:  # Take up to the first 3 keyframes
        desc = kf.prompt.split(".")[0][:60].strip()
        if desc:
            visual_keywords.append(desc)
    combined_visual = ", ".join(visual_keywords)
    music_prompt = f"{config.soundtrack.prompt}. Inspired by: {combined_visual}."
    
    engine = getattr(config.soundtrack, "engine", "local")
    
    # =========================================================================
    # PATH A: ELEVENLABS CLOUD GENERATION (Music v2)
    # =========================================================================
    if engine == "elevenlabs":
        print(f"Generating premium ElevenLabs Music v2 soundtrack: '{music_prompt}'")
        
        # Check if ELEVENLABS_API_KEY is set
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("Error: ELEVENLABS_API_KEY environment variable is not set. Please create a .env file containing ELEVENLABS_API_KEY=your_key in the project directory.")
            
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)
        
        # ElevenLabs supports a maximum of 10 minutes (600,000 ms) in a single request.
        # We generate a track of the requested duration (capped at 10 minutes), and will loop it in FFmpeg if video is longer.
        target_ms = config.video.duration_minutes * 60 * 1000
        music_length_ms = min(600000, target_ms)
        
        print(f"Calling ElevenLabs Music v2 API for composition of {music_length_ms / 1000}s...")
        
        audio_stream = client.music.compose(
            prompt=music_prompt,
            model_id="music_v1",
            music_length_ms=music_length_ms
        )
        
        print(f"Streaming and downloading MP3 chunks to {output_path}...")
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
                
        print(f"ElevenLabs soundtrack generated successfully: {output_path}")
        return

    # =========================================================================
    # PATH B: LOCAL GENERATION (MusicGen)
    # =========================================================================
    print(f"Generating local MusicGen soundtrack: '{music_prompt}'")
    print(f"Loading local MusicGen model '{config.soundtrack.model}' on GPU...")
    
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(config.soundtrack.model)
    model = MusicgenForConditionalGeneration.from_pretrained(
        config.soundtrack.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    sample_rate = model.config.audio_encoder.sampling_rate  # Typically 32000 Hz
    duration_seconds = config.video.duration_minutes * 60
    print(f"MusicGen loaded! Generating {config.video.duration_minutes} minutes ({duration_seconds}s) of audio at {sample_rate}Hz...")
    
    # Generate overlapping segments and crossfade them
    segment_duration = 32
    overlap_duration = 2
    fade_len = int(sample_rate * overlap_duration)
    
    num_segments = int(np.ceil(duration_seconds / 30.0))
    full_audio = []
    
    # Linear crossfade curves
    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in = np.linspace(0.0, 1.0, fade_len)
    
    inputs = processor(
        text=[music_prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    max_tokens = int(round(segment_duration * 50))
    
    with torch.no_grad():
        for s in tqdm(range(num_segments), desc="Generating soundtrack segments"):
            audio_values = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
            segment_wav = audio_values[0, 0].cpu().clamp(-1.0, 1.0).numpy()
            
            if s == 0:
                full_audio = segment_wav
            else:
                wav1_fade = full_audio[-fade_len:] * fade_out
                wav2_fade = segment_wav[:fade_len] * fade_in
                blended = wav1_fade + wav2_fade
                
                full_audio = np.concatenate([
                    full_audio[:-fade_len],
                    blended,
                    segment_wav[fade_len:]
                ])
                
            del audio_values
            if device == "cuda":
                torch.cuda.empty_cache()
                
    trim_len = int(sample_rate * duration_seconds)
    full_audio = full_audio[:trim_len]
    if len(full_audio) > fade_len:
        full_audio[-fade_len:] = full_audio[-fade_len:] * fade_out
        
    temp_wav_path = output_path.replace(".mp3", "_temp.wav")
    print(f"Saving uncompressed audio to {temp_wav_path}...")
    save_wav_file(temp_wav_path, full_audio, sample_rate)
    
    print(f"Compressing soundtrack to MP3 using FFmpeg...")
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_wav_path,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        output_path
    ]
    
    os.system(" ".join(ffmpeg_cmd))
    
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)
        
    print(f"Local soundtrack generated successfully: {output_path}")
