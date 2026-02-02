"""Main Safety Pipeline orchestrator implementing Defense-in-Depth architecture."""

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from config import settings
from models.safety_agent import SafetyAgent, SafetyCheckResult
from models.diffusion import DiffusionGenerator, GenerationResult
from models.coca_embedder import CoCaEmbedder, CoCaResult
from models.clip_embedder import CLIPEmbedder, CLIPCheckResult
from models.nudenet_checker import NudeNetChecker, NudeNetResult
from models.vlm_checker import VLMChecker, VLMCheckResult
from utils.image_utils import apply_blur, save_image, save_image_pair
from utils.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)


class SafetyDecision(str, Enum):
    """Final safety decision."""
    SAFE = "safe"
    UNSAFE_BLURRED = "unsafe_blurred"
    BLOCKED = "blocked"


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    prompt: str
    decision: SafetyDecision | None = None
    image: Image.Image | None = None
    output_path: Path | None = None
    original_path: Path | None = None
    blurred_path: Path | None = None
    
    # Layer results
    pre_check: SafetyCheckResult | None = None
    generation: GenerationResult | None = None
    coca_result: CoCaResult | None = None
    clip_result: CLIPCheckResult | None = None
    caption_recheck: SafetyCheckResult | None = None
    nudenet_result: NudeNetResult | None = None
    vlm_result: VLMCheckResult | None = None
    
    # Timing
    timings: dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0
    
    # Stage status tracking
    stage_status: dict[str, str] = field(default_factory=dict)


class SafetyPipeline:
    """Multi-layered safety pipeline for text-to-image generation.
    
    Implements Defense-in-Depth architecture with:
    1. Pre-generation: LLM agent prompt safety check
    2. Generation: Stable Diffusion (safety checker disabled)
    3. Post-generation: CoCa captioning + Agent re-check
    4. Post-generation: NudeNet + VLM safety analysis
    5. Decision: Safe output / Blur / Block
    """
    
    def __init__(
        self,
        mode: Literal["full", "baseline", "clip_only"] = "full",
        lazy_load: bool = True,
    ):
        self.mode = mode
        self.lazy_load = lazy_load
        
        # Initialize components (lazy loaded)
        self.agent = SafetyAgent()
        self.diffusion = DiffusionGenerator()
        self.coca = CoCaEmbedder()
        self.clip = CLIPEmbedder()
        self.nudenet = NudeNetChecker()
        self.vlm = VLMChecker()
        
        # Metrics
        self.metrics = BenchmarkMetrics()
        
        if not lazy_load:
            self._load_all()
    
    def _load_all(self) -> None:
        """Load all models into memory."""
        self.agent.load()
        self.diffusion.load()
        
        if self.mode == "full":
            self.coca.load()
            self.nudenet.load()
            self.vlm.load()
        elif self.mode == "baseline" or self.mode == "clip_only":
            self.clip.load()
    
    def _unload_all(self) -> None:
        """Unload all models to free VRAM."""
        self.agent.unload()
        self.diffusion.unload()
        self.coca.unload()
        self.clip.unload()
        self.nudenet.unload()
        self.vlm.unload()
        self._free_memory()
    
    def _free_memory(self) -> None:
        """Force garbage collection and clear CUDA cache."""
        if settings.pipeline.force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _unload_model(self, model) -> None:
        """Unload a single model if unload_after_use is enabled."""
        if settings.pipeline.unload_after_use:
            model.unload()
            self._free_memory()
    
    def run(
        self,
        prompt: str,
        seed: int | None = None,
        save_output: bool = True,
    ) -> PipelineResult:
        """Run the complete safety pipeline."""
        start_time = time.perf_counter()
        cfg = settings.pipeline  # Shortcut for pipeline config
        
        result = PipelineResult(prompt=prompt)
        
        # === Stage 1: Pre-generation safety check ===
        if cfg.enable_pre_check:
            t0 = time.perf_counter()
            pre_check = self.agent.check_prompt(prompt)
            result.pre_check = pre_check
            result.timings["pre_check"] = (time.perf_counter() - t0) * 1000
            self._unload_model(self.agent)
            
            if pre_check.is_safe:
                result.stage_status["pre_check"] = "PASSED"
            else:
                result.stage_status["pre_check"] = f"BLOCKED ({pre_check.category or 'unsafe'})"
                result.decision = SafetyDecision.BLOCKED
                result.total_time_ms = (time.perf_counter() - start_time) * 1000
                self._record_metrics(result)
                return result
        else:
            result.stage_status["pre_check"] = "SKIPPED"
        
        # === Stage 2: Image generation ===
        t0 = time.perf_counter()
        gen_result = self.diffusion.generate(prompt=prompt, seed=seed)
        result.generation = gen_result
        result.image = gen_result.image
        result.timings["generation"] = (time.perf_counter() - t0) * 1000
        result.stage_status["generation"] = "COMPLETED"
        self._unload_model(self.diffusion)
        
        # === Stage 3: Post-generation analysis ===
        post_safe = True
        
        if self.mode == "full":
            # CoCa captioning + Agent re-check
            if cfg.enable_coca:
                t0 = time.perf_counter()
                coca_result = self.coca.analyze(gen_result.image)
                result.coca_result = coca_result
                result.timings["coca"] = (time.perf_counter() - t0) * 1000
                result.stage_status["coca"] = "COMPLETED"
                self._unload_model(self.coca)
                
                # Re-check caption with agent
                if cfg.enable_pre_check:
                    t0 = time.perf_counter()
                    caption_check = self.agent.check_caption(coca_result.caption)
                    result.caption_recheck = caption_check
                    result.timings["caption_recheck"] = (time.perf_counter() - t0) * 1000
                    self._unload_model(self.agent)
                    
                    if caption_check.is_safe:
                        result.stage_status["caption_recheck"] = "PASSED"
                    else:
                        result.stage_status["caption_recheck"] = f"CAUGHT ({caption_check.category or 'unsafe'})"
                        post_safe = False
            else:
                result.stage_status["coca"] = "SKIPPED"
                result.stage_status["caption_recheck"] = "SKIPPED"
            
            # NudeNet check
            if cfg.enable_nudenet:
                t0 = time.perf_counter()
                nudenet_result = self.nudenet.check(gen_result.image)
                result.nudenet_result = nudenet_result
                result.timings["nudenet"] = (time.perf_counter() - t0) * 1000
                # NudeNet is lightweight, no unload needed
                
                if nudenet_result.is_nsfw:
                    classes = ", ".join(nudenet_result.triggered_classes[:2])
                    result.stage_status["nudenet"] = f"CAUGHT ({classes})"
                    post_safe = False
                else:
                    result.stage_status["nudenet"] = "PASSED"
            else:
                result.stage_status["nudenet"] = "SKIPPED"
            
            # VLM check
            if cfg.enable_vlm:
                t0 = time.perf_counter()
                vlm_result = self.vlm.analyze(gen_result.image)
                result.vlm_result = vlm_result
                result.timings["vlm"] = (time.perf_counter() - t0) * 1000
                self._unload_model(self.vlm)
                
                if vlm_result.is_safe:
                    result.stage_status["vlm"] = "PASSED"
                else:
                    result.stage_status["vlm"] = f"CAUGHT ({vlm_result.reason[:30]})"
                    post_safe = False
            else:
                result.stage_status["vlm"] = "SKIPPED"
        
        elif self.mode in ("baseline", "clip_only"):
            # CLIP-based safety check (baseline)
            if cfg.enable_clip:
                t0 = time.perf_counter()
                clip_result = self.clip.check_safety(gen_result.image)
                result.clip_result = clip_result
                result.timings["clip"] = (time.perf_counter() - t0) * 1000
                self._unload_model(self.clip)
                
                if clip_result.is_safe:
                    result.stage_status["clip"] = "PASSED"
                else:
                    result.stage_status["clip"] = f"CAUGHT ({clip_result.matched_concept})"
                    post_safe = False
            else:
                result.stage_status["clip"] = "SKIPPED"
        
        # === Stage 4: Final decision ===
        if post_safe:
            result.decision = SafetyDecision.SAFE
            result.stage_status["final"] = "SAFE"
        else:
            result.decision = SafetyDecision.UNSAFE_BLURRED
            result.stage_status["final"] = "UNSAFE_BLURRED"
        
        # Save output
        if save_output:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{timestamp}.png"
            
            if result.decision == SafetyDecision.SAFE:
                # Save only to safe folder
                result.output_path = save_image(gen_result.image, filename=base_name, subfolder="safe")
                result.original_path = result.output_path
            else:
                # Save both original and blurred with same name
                blurred_image = apply_blur(gen_result.image)
                result.original_path, result.blurred_path = save_image_pair(
                    original=gen_result.image,
                    blurred=blurred_image,
                    base_name=base_name,
                )
                result.output_path = result.blurred_path
                result.image = blurred_image
        
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        self._record_metrics(result)
        
        return result
    
    def _record_metrics(self, result: PipelineResult) -> None:
        """Record metrics from pipeline run."""
        if not settings.benchmark_mode:
            return
        
        for layer_name, latency in result.timings.items():
            self.metrics.record_layer_time(layer_name, latency)
        
        self.metrics.record_pipeline_run(result.total_time_ms)
    
    def run_batch(
        self,
        prompts: list[str],
        seeds: list[int] | None = None,
        save_output: bool = True,
    ) -> list[PipelineResult]:
        """Run pipeline on multiple prompts."""
        seeds = seeds or [None] * len(prompts)
        results = []
        
        for prompt, seed in zip(prompts, seeds):
            result = self.run(prompt=prompt, seed=seed, save_output=save_output)
            results.append(result)
        
        return results
    
    def get_metrics_summary(self) -> dict:
        """Get benchmark metrics summary."""
        return self.metrics.get_summary()
    
    def reset_metrics(self) -> None:
        """Reset benchmark metrics."""
        self.metrics.reset()
