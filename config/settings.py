"""Configuration settings for the NSFW Safety Filtering Pipeline."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model identifiers and paths."""
    
    # Pre-generation Agent (LLM)
    safety_agent_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    safety_agent_quantization: Literal["4bit", "8bit", "none"] = "4bit"
    
    # Stable Diffusion
    diffusion_id: str = "runwayml/stable-diffusion-v1-5"
    diffusion_dtype: str = "float16"
    
    # CLIP (baseline)
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    
    # CoCa
    coca_model: str = "coca_ViT-L-14"
    coca_pretrained: str = "laion2b_s13b_b90k"
    
    # VLM
    vlm_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_quantization: Literal["4bit", "8bit", "none"] = "4bit"


class ThresholdConfig(BaseModel):
    """Safety thresholds for different components."""
    
    # NudeNet thresholds
    nudenet_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    nudenet_unsafe_classes: list[str] = [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED", 
        "MALE_GENITALIA_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    ]
    
    # CLIP/CoCa similarity threshold
    embedding_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    
    # VLM confidence threshold
    vlm_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Pipeline stage toggles for flexible testing."""
    
    # Stage toggles
    enable_pre_check: bool = True       # SafetyAgent LLM pre-check
    enable_coca: bool = True            # CoCa captioning + re-check
    enable_clip: bool = True            # CLIP baseline check
    enable_nudenet: bool = True         # NudeNet nudity detection
    enable_vlm: bool = True            # VLM semantic analysis (disabled by default - high VRAM)
    
    # Memory management
    unload_after_use: bool = True       # Unload each model after use to save VRAM
    force_gc: bool = True               # Force garbage collection after unload


class GenerationConfig(BaseModel):
    """Stable Diffusion generation parameters."""
    
    num_inference_steps: int = 70
    guidance_scale: float = 4
    width: int = 512
    height: int = 512
    enable_xformers: bool = True
    safety_checker: bool = False  # Disabled for our custom pipeline


class PathConfig(BaseModel):
    """File paths and directories."""
    
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    model_cache: Path = Path(".cache/models")
    
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        for path in [self.output_dir, self.cache_dir, self.model_cache]:
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Main settings container."""
    
    models: ModelConfig = ModelConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    generation: GenerationConfig = GenerationConfig()
    pipeline: PipelineConfig = PipelineConfig()
    paths: PathConfig = PathConfig()
    
    # Device settings
    device: str = "cuda"
    
    # Benchmark mode
    benchmark_mode: bool = False
    
    # NSFW categories for agent prompt
    nsfw_categories: list[str] = [
        "sexual",
        "nudity", 
        "violence",
        "gore",
        "hate",
        "harassment",
        "self-harm",
        "illegal_activity",
        "shocking",
    ]
    
    # Blur settings
    blur_kernel_size: int = 99
    blur_sigma: float = 30.0


# Global settings instance
settings = Settings()
