# Atmogen

Atmogen is a CLI tool designed to generate high-fidelity, long-duration (1-2 hours), slow-morphing atmospheric background videos. It combines cloud-based generative AI for keyframes, local GPU-accelerated frame interpolation, and AI-generated soundtracks to create seamless visual experiences at 4K resolution.

## 🚀 Core Features

- **Cloud-Native Keyframing**: Leverages FLUX.1 via Fal.ai to generate high-quality atmospheric anchors and transitions.
- **Pristine Morphing Pipeline**: Uses a "2-Path Reference Buffer" that preserves native 2K frames for image-to-image chaining, preventing style drift and concept flipping across long sequences.
- **GPU-Accelerated VFI**: Integrates RIFE (Real-Time Intermediate Flow Estimation) to transform sparse AI keyframes into fluid high-fps video.
- **Native 4K Upscaling**: Implements a Swin2SR pipeline to upscale frames from generative resolutions (1024x576) to full 4K (3840x2160).
- **AI Soundtracks**: Generates custom ambient audio via ElevenLabs Music or local MusicGen, which is then looped infinitely to match any video duration.
- **Video Extension Mode**: Can take an existing input video, upscale it, and stretch its temporal length using RIFE interpolation.

## 🛠 Architecture & Design Decisions

### The "Pristine Buffer" Strategy
To avoid the common AI problem of "concept decay" (where a scene slowly morphs into noise or cartoons over time), Atmogen utilizes a chained `img2img` process. It maintains native resolution references for every frame in the chain, ensuring that each new frame is based on the original's structural integrity rather than a degraded upscaled version.

### Hardware Optimization (DGX Spark)
The tool is optimized for high-VRAM environments but incorporates critical memory safety measures to prevent kernel panics:
- **RIFE Memory Capping**: The RIFE tensor queue is limited to `maxsize=8` and implements streaming buffers to prevent VRAM spikes during the interpolation of thousands of frames.
- **Cloud Offloading**: Keyframe generation is offloaded to Fal.ai (FLUX.1) to reduce local compute costs and memory pressure on the GPU while maintaining state-of-the-art visual quality.

### Temporal Stretching
For the `extend` functionality, Atmogen extract all native frames from a source video, upscales them, and calculates a RIFE multiplier based on the ratio between the source frame count and the target duration (Duration $\times$ FPS). This allows short clips to be stretched into atmospheric loops without losing original detail.

## 📦 Installation & Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- FFmpeg installed in system PATH
- Python 3.10+

### Configuration
Create a `.env` file in the project root:
```env
FAL_KEY=your_fal_ai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

## 🖥 Usage

### Preview Keyframes
Before committing to a full render, generate preview images of the target keyframes defined in your YAML config:
```bash
python main.py preview configs/gem_0.yaml
```

### Full Render
Run the end-to-end pipeline: Sparse Frame Generation $\rightarrow$ Upscaling $\rightarrow$ RIFE Interpolation $\rightarrow$ Soundtrack Muxing.
```bash
python main.py render configs/gem_0.yaml
```

### Extend Existing Video
Upscale a source video and stretch its duration to match the config settings:
```bash
python main.py extend input/source_video.mp4 configs/gem_0.yaml
```

### Generate Soundtrack Only
Generate just the AI ambient audio for a specific configuration:
```bash
python main.py soundtrack configs/gem_0.yaml
```

## 📄 Configuration (YAML)
Configs are defined in YAML files (e.g., `gem_*.yaml`). Key parameters include:
- `video`: Target resolution, duration, and output FPS.
- `keyframes`: A timeline of prompts and seeds to guide the morphing process.
- `soundtrack`: AI engine choice (local/elevenlabs) and mood prompt.
- `flux_img2img`: Boolean to enable chained transitions between frames for smoother auras.
