# Atmogen Project Notes

## Intent
Atmogen is a CLI tool designed to generate extremely long (1 to 2 hour), slow-morphing atmospheric background videos. The core concept is to create a static-camera video where the environment undergoes imperceptible changes over a long period (e.g., a post-apocalyptic city where the buildings very slowly morph into towering organic creatures). 

Because the changes happen so slowly, generating every single frame at 30fps using an AI model would take immense compute time. The solution is **sparse generation with heavy interpolation**.

## Architecture & Workflow
The tool runs in a two-step process to save compute and give the user creative control. It is built in Python and relies heavily on HuggingFace Diffusers (Stable Diffusion XL) and FFmpeg.

### 1. Configuration (`example_config.yaml`)
The user defines the timeline, resolution, and specific anchors (keyframes).
Each keyframe has a specific timestamp (e.g., "00:30:00"), a text prompt, and a specific seed.

### 2. Preview Phase (`python main.py preview example_config.yaml`)
*   **Goal:** Allow the user to review the anchor frames without rendering the whole video.
*   **Action:** The software boots up SDXL and generates *only* the specific images defined in the keyframes section. It saves them to `./output/preview/`.
*   **Iterate:** The user can tweak prompts and seeds until they are happy with the exact look of each keyframe.

### 3. Render Phase (`python main.py render example_config.yaml`)
*   **Goal:** Generate the final morphing video.
*   **A. Sparse Generation:** The tool generates 1 frame every 5 seconds (configurable via `sparse_fps`). 
    *   To get perfectly smooth transitions between Keyframe A and Keyframe B, we use mathematical interpolation.
    *   **Latent Slerp (Spherical Linear Interpolation):** We transition the initial noise (seed) smoothly.
    *   **Embedding Lerp (Linear Interpolation):** We transition the text prompt embeddings smoothly.
    *   The model runs on this interpolated data, generating a frame that is a mathematical halfway point between the two concepts.
*   **B. Video Compilation & Frame Interpolation:** 
    *   Once the sparse frames are generated, we use `ffmpeg` with `minterpolate` to turn the sparse framerate (e.g., 0.2 fps) into a smooth output framerate (e.g., 30 fps). 
    *   FFmpeg hallucinates the missing frames using motion estimation. (Note: For future iterations, a dedicated AI interpolation tool like RIFE or FILM could replace the FFmpeg minterpolate for even better quality, but minterpolate is a great built-in starting point).

## Current State
*   The core scaffolding is complete.
*   `config.py` handles parsing the YAML timeline.
*   `generator.py` handles the SDXL PyTorch logic (preview generation, latent slerp, embedding lerp).
*   `video.py` handles the FFmpeg compilation.
*   `cli.py` handles the CLI commands.
*   `setup.sh` is provided for easy virtual environment creation and dependency installation.

## Environment Details
*   **Target Machine:** DGX Spark (High VRAM/Compute available).
*   **Model:** Stable Diffusion XL 1.0 (FP16).
*   **Camera:** Static.
*   **Audio:** None for this proof-of-concept phase.

## Future Ideas / Next Steps
*   If the FFmpeg `minterpolate` struggles with complex generative morphs, consider implementing a dedicated python-based RIFE (Real-Time Intermediate Flow Estimation) pipeline for step 3.B.
*   Implement automatic color correction (histogram matching) if the autoregressive steps cause the images to become "deep-fried" or over-contrasted across long generations. (Less likely with the Slerp approach, but worth keeping an eye on).
*   Potentially add an audio overlay step in FFmpeg.
*   Eventually, a Web UI (Gradio/Streamlit) could be built on top of this.