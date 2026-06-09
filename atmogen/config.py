import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Keyframe:
    time_str: str
    time_seconds: int
    prompt: str
    seed: int

@dataclass
class VideoConfig:
    width: int
    height: int
    duration_minutes: int
    output_fps: int
    sparse_fps: float

@dataclass
class ControlNetConfig:
    enabled: bool
    type: str  # 'depth' or 'canny'
    control_strength: float

@dataclass
class SoundtrackConfig:
    enabled: bool
    model: str
    prompt: str
    engine: str = "local"

@dataclass
class AtmogenConfig:
    video: VideoConfig
    keyframes: list[Keyframe]
    output_dir: str
    controlnet: ControlNetConfig = None
    engine: str = "local"
    flux_all_sparse: bool = False
    flux_img2img: bool = False
    flux_denoise_strength: float = 0.3
    soundtrack: SoundtrackConfig = None

def load_env():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def parse_time_to_seconds(time_str: str) -> int:
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def load_config(file_path: str) -> AtmogenConfig:
    load_env()
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    v_data = data.get('video', {})
    video_config = VideoConfig(
        width=v_data.get('width', 1024),
        height=v_data.get('height', 576),
        duration_minutes=v_data.get('duration_minutes', 60),
        output_fps=v_data.get('output_fps', 30),
        sparse_fps=v_data.get('sparse_fps', 0.2)
    )

    k_data = data.get('keyframes', {})
    keyframes = []
    for t_str, k_info in k_data.items():
        time_seconds = parse_time_to_seconds(t_str)
        keyframes.append(Keyframe(
            time_str=t_str,
            time_seconds=time_seconds,
            prompt=k_info['prompt'],
            seed=k_info['seed']
        ))
    
    # Sort keyframes by time
    keyframes.sort(key=lambda k: k.time_seconds)
    
    output_dir = data.get('output_dir') or v_data.get('output_dir')
    if not output_dir:
        base_name = os.path.basename(file_path)
        output_dir = os.path.splitext(base_name)[0]
    
    # Parse ControlNet config
    c_data = data.get('controlnet', {})
    controlnet_config = ControlNetConfig(
        enabled=c_data.get('enabled', False),
        type=c_data.get('type', 'depth'),
        control_strength=c_data.get('control_strength', 0.7)
    )
    
    engine = data.get('engine', 'local')
    flux_all_sparse = data.get('flux_all_sparse') or v_data.get('flux_all_sparse', False)
    
    # Parse flux_img2img settings
    flux_img2img = data.get('flux_img2img', False)
    flux_denoise_strength = 0.3
    if isinstance(flux_img2img, dict):
        flux_denoise_strength = flux_img2img.get('denoise_strength', 0.3)
        flux_img2img = flux_img2img.get('enabled', False)
    else:
        flux_denoise_strength = data.get('flux_denoise_strength', 0.3)
        
    # Parse soundtrack config
    s_data = data.get('soundtrack', {})
    soundtrack_config = SoundtrackConfig(
        enabled=s_data.get('enabled', False),
        model=s_data.get('model', 'facebook/musicgen-medium'),
        prompt=s_data.get('prompt', 'slow deep dark industrial drone, low ambient chthonic pads, gothic, horror'),
        engine=s_data.get('engine', 'local')
    )
    
    return AtmogenConfig(
        video=video_config,
        keyframes=keyframes,
        output_dir=output_dir,
        controlnet=controlnet_config,
        engine=engine,
        flux_all_sparse=bool(flux_all_sparse),
        flux_img2img=bool(flux_img2img),
        flux_denoise_strength=float(flux_denoise_strength),
        soundtrack=soundtrack_config
    )
