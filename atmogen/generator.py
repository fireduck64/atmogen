import os
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm
from .config import AtmogenConfig, Keyframe
from .slerp import slerp

class VideoGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.upscaler_model = None
        self.upscaler_processor = None
        self.controlnet_pipeline = None
        self.depth_estimator = None

    def load_pipeline(self):
        if self.pipeline is None:
            print(f"Loading SDXL Pipeline to {self.device}...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True
            )
            self.pipeline.to(self.device)
            # Optional: self.pipeline.enable_model_cpu_offload() or self.pipeline.enable_xformers_memory_efficient_attention()
            print("Pipeline loaded.")

    def generate_fal_image(self, prompt: str, seed: int, width: int, height: int) -> Image.Image:
        import fal_client
        import requests
        from io import BytesIO
        
        fal_w = width
        fal_h = height
        if fal_w > 2048 or fal_h > 2048:
            scale = min(2048 / fal_w, 2048 / fal_h)
            fal_w = int(round(fal_w * scale / 8) * 8)
            fal_h = int(round(fal_h * scale / 8) * 8)
            
        print(f"Calling Fal.ai FLUX 1.1 [pro] ultra API at {fal_w}x{fal_h}...")
        
        # Check if FAL_KEY is set
        if not os.environ.get("FAL_KEY"):
            raise ValueError("Error: FAL_KEY environment variable is not set. Please create a .env file containing FAL_KEY=your_key in the project directory.")
            
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1-ultra",
            arguments={
                "prompt": prompt,
                "image_size": {"width": fal_w, "height": fal_h},
                "seed": seed,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "enable_safety_checker": False
            }
        )
        
        image_url = result["images"][0]["url"]
        print(f"Image generated! Downloading from {image_url}")
        
        resp = requests.get(image_url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.size != (fal_w, fal_h):
            img = img.resize((fal_w, fal_h), Image.Resampling.LANCZOS)
        return img

    def generate_fal_img2img_image(self, previous_image_path: str, prompt: str, seed: int, strength: float, width: int, height: int) -> Image.Image:
        import fal_client
        import requests
        from io import BytesIO
        
        fal_w = width
        fal_h = height
        if fal_w > 2048 or fal_h > 2048:
            scale = min(2048 / fal_w, 2048 / fal_h)
            fal_w = int(round(fal_w * scale / 8) * 8)
            fal_h = int(round(fal_h * scale / 8) * 8)
            
        print(f"Uploading previous frame {previous_image_path} to Fal.ai...")
        # Check FAL_KEY
        if not os.environ.get("FAL_KEY"):
            raise ValueError("Error: FAL_KEY environment variable is not set. Please create a .env file containing FAL_KEY=your_key in the project directory.")
            
        # Upload file directly via fal_client
        image_url = fal_client.upload_file(previous_image_path)
        print(f"Uploaded! File URL: {image_url}")
        
        print(f"Calling Fal.ai FLUX [dev] Image-to-Image API at {fal_w}x{fal_h} with denoise strength={strength}...")
        
        result = fal_client.subscribe(
            "fal-ai/flux/dev/image-to-image",
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "image_size": {"width": fal_w, "height": fal_h},
                "strength": strength,
                "seed": seed,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "enable_safety_checker": False
            }
        )
        
        output_url = result["images"][0]["url"]
        print(f"Image generated! Downloading from {output_url}")
        
        resp = requests.get(output_url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.size != (fal_w, fal_h):
            img = img.resize((fal_w, fal_h), Image.Resampling.LANCZOS)
        return img

    def load_upscaler(self):
        if self.upscaler_model is None:
            print("Loading Swin2SR upscaler to device...")
            from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
            self.upscaler_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
            self.upscaler_model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64").to(self.device)
            print("Upscaler loaded.")

    def upscale_image(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        self.load_upscaler()
        inputs = self.upscaler_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.upscaler_model(**inputs)
        
        output_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp(0, 1).numpy()
        output_tensor = np.moveaxis(output_tensor, 0, -1)
        output_img = Image.fromarray(np.uint8(output_tensor * 255))
        
        if output_img.size != (target_width, target_height):
            output_img = output_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return output_img

    def load_controlnet_pipeline(self):
        if self.controlnet_pipeline is None:
            self.load_pipeline()
            print("Loading ControlNet model...")
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            print("Creating ControlNet Pipeline sharing weights with base pipeline...")
            self.controlnet_pipeline = StableDiffusionXLControlNetPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                unet=self.pipeline.unet,
                controlnet=controlnet,
                scheduler=self.pipeline.scheduler
            )
            self.controlnet_pipeline.to(self.device)
            print("ControlNet Pipeline loaded.")

    def load_depth_estimator(self):
        if self.depth_estimator is None:
            print("Loading Depth Estimator (depth-anything-small)...")
            from transformers import pipeline
            self.depth_estimator = pipeline(
                "depth-estimation", 
                model="LiheYoung/depth-anything-small-hf",
                device=0 if self.device == "cuda" else -1
            )
            print("Depth Estimator loaded.")

    def extract_depth_map(self, image: Image.Image) -> Image.Image:
        self.load_depth_estimator()
        depth_output = self.depth_estimator(image)
        return depth_output["depth"].resize(image.size)

    def get_generation_dimensions(self, config: AtmogenConfig):
        target_w = config.video.width
        target_h = config.video.height
        if target_w > 1024 or target_h > 1024:
            scale = min(1024 / target_w, 1024 / target_h)
            gen_w = int(round(target_w * scale / 8) * 8)
            gen_h = int(round(target_h * scale / 8) * 8)
            return gen_w, gen_h, True
        return target_w, target_h, False

    def generate_preview(self, config: AtmogenConfig, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        gen_w, gen_h, should_upscale = self.get_generation_dimensions(config)
        if should_upscale:
            print(f"Target resolution {config.video.width}x{config.video.height} is large. Generating at native {gen_w}x{gen_h} and upscaling...")
        
        use_controlnet = config.controlnet and config.controlnet.enabled
        depth_map = None
        
        for idx, kf in enumerate(config.keyframes):
            print(f"Generating preview for {kf.time_str} (Seed: {kf.seed})...")
            generator = torch.Generator(device=self.device).manual_seed(kf.seed)
            
            if idx == 0 or not use_controlnet:
                # First frame is generated natively or via Fal.ai without ControlNet to establish the anchor
                if config.engine == "fal-flux":
                    image = self.generate_fal_image(kf.prompt, kf.seed, config.video.width, config.video.height)
                    if use_controlnet and image.size != (gen_w, gen_h):
                        image = image.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
                else:
                    self.load_pipeline()
                    image = self.pipeline(
                        prompt=kf.prompt,
                        generator=generator,
                        width=gen_w,
                        height=gen_h,
                        num_inference_steps=40
                    ).images[0]
                
                if use_controlnet:
                    print("Extracting depth map from the anchor frame...")
                    depth_map = self.extract_depth_map(image)
                    depth_save_path = os.path.join(output_dir, "depth_anchor.png")
                    depth_map.save(depth_save_path)
                    print(f"Saved depth anchor visualization to {depth_save_path}")
            else:
                # Subsequent frames use ControlNet with the depth map of the anchor
                self.load_controlnet_pipeline()
                image = self.controlnet_pipeline(
                    prompt=kf.prompt,
                    image=depth_map,
                    controlnet_conditioning_scale=config.controlnet.control_strength,
                    generator=generator,
                    width=gen_w,
                    height=gen_h,
                    num_inference_steps=40
                ).images[0]
            
            if should_upscale:
                image = self.upscale_image(image, config.video.width, config.video.height)
                
            safe_time = kf.time_str.replace(":", "-")
            out_path = os.path.join(output_dir, f"preview_{safe_time}.png")
            image.save(out_path)
            print(f"Saved {out_path}")

    def get_prompt_embeddings(self, prompt: str):
        # Obtain text embeddings
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="blurry, low quality, artifact, deformed"
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def generate_render(self, config: AtmogenConfig, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        duration_seconds = config.video.duration_minutes * 60
        num_sparse_frames = int(duration_seconds * config.video.sparse_fps)
        time_per_frame = 1.0 / config.video.sparse_fps
        
        # Check if all sparse frames already exist on disk
        all_exist = True
        for i in range(num_sparse_frames):
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            if not os.path.exists(frame_path):
                all_exist = False
                break
                
        if all_exist and num_sparse_frames > 0:
            print(f"All {num_sparse_frames} sparse frames already exist in {output_dir}. Skipping generation phase to save API costs and time!")
            return
            
        gen_w, gen_h, should_upscale = self.get_generation_dimensions(config)
        
        # Check if we should generate all sparse frames via FLUX on the cloud
        if config.engine == "fal-flux" and config.flux_all_sparse:
            print(f"Generating ALL {num_sparse_frames} sparse frames using Fal.ai FLUX.1 on the cloud...")
            
            def blend_prompts(prompt1: str, prompt2: str, t: float) -> str:
                percentage_1 = int((1 - t) * 100)
                percentage_2 = int(t * 100)
                return f"A hybrid transitional state, morphing from: ({prompt1}) to: ({prompt2}). The scene is currently {percentage_1}% the first state and {percentage_2}% the second state, capturing a fluid intermediate morphing transformation."

            for i in tqdm(range(num_sparse_frames)):
                current_time = i * time_per_frame
                
                # Find bounding keyframes
                idx = 0
                while idx < len(config.keyframes) - 1 and config.keyframes[idx+1].time_seconds <= current_time:
                    idx += 1
                    
                kf1 = config.keyframes[idx]
                if idx + 1 < len(config.keyframes):
                    kf2 = config.keyframes[idx+1]
                    t = (current_time - kf1.time_seconds) / (kf2.time_seconds - kf1.time_seconds)
                    prompt = blend_prompts(kf1.prompt, kf2.prompt, t)
                    seed = kf1.seed
                else:
                    prompt = kf1.prompt
                    seed = kf1.seed
                
                frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
                native_frame_path = os.path.join(output_dir, f"frame_{i:06d}_native.png")
                
                # Chained img2img logic
                if i > 0 and config.flux_img2img:
                    prev_native_path = os.path.join(output_dir, f"frame_{i-1:06d}_native.png")
                    image = self.generate_fal_img2img_image(
                        prev_native_path,
                        prompt,
                        seed,
                        config.flux_denoise_strength,
                        config.video.width,
                        config.video.height
                    )
                else:
                    image = self.generate_fal_image(prompt, seed, config.video.width, config.video.height)
                
                # Save the pristine 2K native frame
                image.save(native_frame_path)
                
                # Save the upscaled 4K frame for compilation
                if should_upscale:
                    image_4k = self.upscale_image(image, config.video.width, config.video.height)
                    image_4k.save(frame_path)
                else:
                    image.save(frame_path)
            
            # Clean up pristine native frames to save disk space
            print("Cleaning up temporary native reference frames...")
            import glob
            for f in glob.glob(os.path.join(output_dir, "*_native.png")):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing native reference frame: {e}")
            return

        # Defer local pipeline load to here
        self.load_pipeline()
        
        if should_upscale:
            print(f"Target resolution {config.video.width}x{config.video.height} is large. Generating sparse frames at native {gen_w}x{gen_h} and upscaling...")
            
        print(f"Generating {num_sparse_frames} sparse frames into {output_dir}...")
        
        use_controlnet = config.controlnet and config.controlnet.enabled
        depth_map = None
        
        # Precompute embeddings and latents for keyframes
        kf_data = []
        for kf in config.keyframes:
            # Generate initial latents
            shape = (1, self.pipeline.unet.config.in_channels, gen_h // 8, gen_w // 8)
            generator = torch.Generator(device=self.device).manual_seed(kf.seed)
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.pipeline.unet.dtype)
            
            # Get embeddings
            prompt_embeds, neg_embeds, pooled_embeds, neg_pooled_embeds = self.get_prompt_embeddings(kf.prompt)
            
            kf_data.append({
                "time": kf.time_seconds,
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "neg_embeds": neg_embeds,
                "pooled_embeds": pooled_embeds,
                "neg_pooled_embeds": neg_pooled_embeds
            })

        for i in tqdm(range(num_sparse_frames)):
            current_time = i * time_per_frame
            
            # Find the two bounding keyframes
            idx = 0
            while idx < len(kf_data) - 1 and kf_data[idx+1]["time"] <= current_time:
                idx += 1
                
            kf1 = kf_data[idx]
            if idx + 1 < len(kf_data):
                kf2 = kf_data[idx+1]
                # Interpolate
                t = (current_time - kf1["time"]) / (kf2["time"] - kf1["time"])
                
                # Slerp latents
                interp_latents = slerp(t, kf1["latents"].view(-1), kf2["latents"].view(-1)).view(kf1["latents"].shape)
                
                # Lerp embeddings
                interp_prompt_embeds = (1 - t) * kf1["prompt_embeds"] + t * kf2["prompt_embeds"]
                interp_pooled_embeds = (1 - t) * kf1["pooled_embeds"] + t * kf2["pooled_embeds"]
                
                neg_embeds = kf1["neg_embeds"] # keep negative constant or lerp it
                neg_pooled_embeds = kf1["neg_pooled_embeds"]
                
            else:
                # Past the last keyframe, use the last one
                interp_latents = kf1["latents"]
                interp_prompt_embeds = kf1["prompt_embeds"]
                interp_pooled_embeds = kf1["pooled_embeds"]
                neg_embeds = kf1["neg_embeds"]
                neg_pooled_embeds = kf1["neg_pooled_embeds"]

            # Generate image
            if i == 0 or not use_controlnet:
                if i == 0 and config.engine == "fal-flux":
                    image = self.generate_fal_image(config.keyframes[0].prompt, config.keyframes[0].seed, config.video.width, config.video.height)
                    if use_controlnet and image.size != (gen_w, gen_h):
                        image = image.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
                else:
                    image = self.pipeline(
                        prompt_embeds=interp_prompt_embeds,
                        negative_prompt_embeds=neg_embeds,
                        pooled_prompt_embeds=interp_pooled_embeds,
                        negative_pooled_prompt_embeds=neg_pooled_embeds,
                        latents=interp_latents,
                        width=gen_w,
                        height=gen_h,
                        num_inference_steps=40,
                        guidance_scale=7.5
                    ).images[0]
                
                if i == 0 and use_controlnet:
                    print("Extracting depth map from the anchor frame...")
                    depth_map = self.extract_depth_map(image)
                    depth_save_path = os.path.join(output_dir, "depth_anchor.png")
                    depth_map.save(depth_save_path)
                    print(f"Saved depth anchor visualization to {depth_save_path}")
            else:
                self.load_controlnet_pipeline()
                image = self.controlnet_pipeline(
                    prompt_embeds=interp_prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    pooled_prompt_embeds=interp_pooled_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_embeds,
                    image=depth_map,
                    controlnet_conditioning_scale=config.controlnet.control_strength,
                    latents=interp_latents,
                    width=gen_w,
                    height=gen_h,
                    num_inference_steps=40,
                    guidance_scale=7.5
                ).images[0]
            
            if should_upscale:
                image = self.upscale_image(image, config.video.width, config.video.height)
            
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            image.save(frame_path)
