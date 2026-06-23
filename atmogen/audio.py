import os
import sys
import wave
import struct
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

def save_wav_file(filename, samples, sample_rate=32000):
    """
    Saves a 1D float array of audio samples as a 16-bit PCM WAV file using built-in wave module.
    """
    int_samples = [int(max(-32768, min(32767, x * 32767))) for x in samples]
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(int_samples)}h", *int_samples))

def generate_audio_asset(prompt: str, engine: str, model: str = "facebook/musicgen-medium", output_path: str = "output.mp3"):
    """
    Pure generation function that takes a prompt and produces an MP3 file.
    Decoupled from AtmogenConfig to allow standalone asset creation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if engine == "elevenlabs":
        print(f"Generating premium ElevenLabs Music v2 soundtrack: '{prompt}'")
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("Error: ELEVENLABS_API_KEY is not set.")

        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)

        # Asset generation usually targets a standard high-quality length (e.g., 5 mins)
        # as it will be looped in the final video anyway.
        music_length_ms = 600000  # 10 minutes
        print(f"Calling ElevenLabs Music v2 API for composition of {music_length_ms / 1000}s...")

        audio_stream = client.music.compose(
            prompt=prompt,
            model_id="music_v2",
            music_length_ms=music_length_ms
        )

        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return output_path

    elif engine == "local":
        print(f"Generating local MusicGen soundtrack: '{prompt}'")
        print(f"Loading model '{model}' on GPU...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model)
        model_obj = MusicgenForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        sample_rate = model_obj.config.audio_encoder.sampling_rate
        duration_seconds = 300 # Generate a standard 5-minute asset for the library
        print(f"MusicGen loaded! Generating {duration_seconds}s of audio at {sample_rate}Hz...")

        segment_duration = 32
        overlap_duration = 2
        fade_len = int(sample_rate * overlap_duration)
        num_segments = int(np.ceil(duration_seconds / 30.0))
        full_audio = []

        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in = np.linspace(0.0, 1.0, fade_len)

        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        max_tokens = int(round(segment_duration * 50))

        with torch.no_grad():
            for s in tqdm(range(num_segments), desc="Generating audio asset"):
                audio_values = model_obj.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
                segment_wav = audio_values[0, 0].cpu().clamp(-1.0, 1.0).numpy()

                if s == 0:
                    full_audio = segment_wav
                else:
                    wav1_fade = full_audio[-fade_len:] * fade_out
                    wav2_fade = segment_wav[:fade_len] * fade_in
                    blended = wav1_fade + wav2_fade
                    full_audio = np.concatenate([full_audio[:-fade_len], blended, segment_wav[fade_len:]])

                del audio_values
                if device == "cuda": torch.cuda.empty_cache()

        trim_len = int(sample_rate * duration_seconds)
        full_audio = full_audio[:trim_len]
        if len(full_audio) > fade_len:
            full_audio[-fade_len:] = full_audio[-fade_len:] * fade_out

        temp_wav_path = output_path.replace(".mp3", "_temp.wav")
        save_wav_file(temp_wav_path, full_audio, sample_rate)

        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", output_path
        ], capture_output=True)

        if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
        return output_path

    else:
        raise ValueError(f"Unsupported audio engine: {engine}")
