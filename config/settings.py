"""Configuration settings for the NSFW Safety Filtering Pipeline."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model identifiers and paths."""
    
    # Pre-generation Agent (LLM)
    safety_agent_id: str = "Qwen/Qwen2.5-7B-Instruct"
    safety_agent_quantization: Literal["4bit", "8bit", "none"] = "4bit"
    
    # Stable Diffusion
    #diffusion_id: str = "runwayml/stable-diffusion-v1-5"
    #diffusion_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    diffusion_id: str = "stablediffusionapi/deliberate-v5"
    diffusion_dtype: str = "float16"
    
    # CLIP (baseline)
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    
    # CoCa
    coca_model: str = "coca_ViT-L-14"
    coca_pretrained: str = "laion2b_s13b_b90k"
    
    # VLM
    vlm_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    #vlm_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    #vlm_id: str = "llava-hf/llama3-llava-next-8b-hf"
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
    
    # CLIP global fallback threshold (per-class tanımlanmamışsa kullanılır)
    embedding_threshold: float = Field(default=0.28, ge=0.0, le=1.0)

    # CLIP per-class threshold'lar — her kavram için ayrı eşik
    # Referans: Rando et al. (2022) Table 1 — SD v1.4 safety filter orijinal threshold'ları
    # Cinsel içerik: 0.18–0.22 (SD orijinali), Şiddet/gore: 0.26–0.30 (false positive riski yüksek)
    clip_per_class_thresholds: dict[str, float] = {
        # Cinsel içerik — SD orijinal threshold'larına yakın (0.18–0.22)
        "sexual":           0.18,
        "nude":             0.19,
        "sex":              0.20,
        "naked":            0.19,
        "nsfw":             0.19,
        "porn":             0.19,
        "18+":              0.21,
        "explicit content": 0.19,
        "uncensored":       0.19,
        "nipples":          0.21,
        "naked breast":     0.21,
        "vagina":           0.19,
        "areola":           0.20,
        "naked person":     0.22,
        # Şiddet / gore — daha yüksek (aksiyon/korku sahneleri false positive verebilir)
        "gore":             0.19,
        "violence":         0.20,
        "bloody":           0.18,
        "disturbing":       0.20,
        "shocking":         0.21,
        "horrifying":       0.20,
        # Çocuk koruması — SD special care threshold'ları (agresif, düşük)
        "young child":      0.19,
        "young girl":       0.20,
    }

    # VLM confidence threshold
    vlm_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Pipeline stage toggles for flexible testing."""
    
    # Stage toggles — her biri bağımsız olarak açılıp kapatılabilir
    enable_pre_check: bool = True       # SafetyAgent LLM pre-check
    enable_coca: bool = True            # CoCa captioning + re-check
    enable_clip: bool = False           # CLIP similarity check (default kapalı, True yapınca aktif)
    enable_nudenet: bool = True         # NudeNet nudity detection
    enable_vlm: bool = True             # VLM semantic analysis
    
    # Memory management
    unload_after_use: bool = True       # Unload each model after use to save VRAM
    force_gc: bool = True               # Force garbage collection after unload

    @classmethod
    def preset_full(cls) -> "PipelineConfig":
        """Agent + CoCa + NudeNet + VLM (varsayılan, maksimum güvenlik)."""
        return cls(
            enable_pre_check=True,
            enable_coca=True,
            enable_nudenet=True,
            enable_vlm=True,
            enable_clip=False,
        )

    @classmethod
    def preset_baseline(cls) -> "PipelineConfig":
        """Agent + CLIP (hızlı, temel güvenlik karşılaştırması)."""
        return cls(
            enable_pre_check=True,
            enable_clip=True,
            enable_coca=False,
            enable_nudenet=False,
            enable_vlm=False,
        )

    @classmethod
    def preset_clip_only(cls) -> "PipelineConfig":
        """Sadece CLIP (pre-check yok, minimum filtreleme)."""
        return cls(
            enable_pre_check=False,
            enable_clip=True,
            enable_coca=False,
            enable_nudenet=False,
            enable_vlm=False,
        )


class GenerationConfig(BaseModel):
    """Stable Diffusion generation parameters."""
    
    num_inference_steps: int = 50
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
