"""Stable Diffusion wrapper with safety checker disabled."""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of image generation."""
    image: Image.Image
    seed: int
    prompt: str
    guidance_scale: float = 0.0
    width: int = 0
    height: int = 0
    num_steps: int = 0


class DiffusionGenerator:
    """Stable Diffusion 1.5 image generator with safety checker disabled."""
    
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or settings.models.diffusion_id
        self.device = device or settings.device
        self.pipe = None
        self._loaded = False
    
    def load(self) -> None:
        """Load Stable Diffusion pipeline."""
        if self._loaded:
            return
        
        # Determine dtype
        dtype = torch.float16 if settings.models.diffusion_dtype == "float16" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disabled - we use our own pipeline
            requires_safety_checker=False,
        )
        
        # Use DPM-Solver++ for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable xformers for memory efficiency
        if settings.generation.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"xformers not available: {e}")
        
        # Enable attention slicing as fallback
        self.pipe.enable_attention_slicing()
        
        self._loaded = True
    
    def unload(self) -> None:
        """Unload pipeline to free VRAM."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        torch.cuda.empty_cache()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> GenerationResult:
        """Generate an image from prompt."""
        if not self._loaded:
            self.load()
        
        # Use settings defaults
        num_steps = num_steps or settings.generation.num_inference_steps
        guidance_scale = guidance_scale or settings.generation.guidance_scale
        width = width or settings.generation.width
        height = height or settings.generation.height
        
        # Handle seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
        
        return GenerationResult(
            image=result.images[0],
            seed=seed,
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_steps=num_steps,
        )
    
    def generate_batch(
        self,
        prompts: list[str],
        seeds: list[int] | None = None,
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate multiple images."""
        results = []
        seeds = seeds or [None] * len(prompts)
        
        for prompt, seed in zip(prompts, seeds):
            result = self.generate(prompt=prompt, seed=seed, **kwargs)
            results.append(result)
        
        return results
