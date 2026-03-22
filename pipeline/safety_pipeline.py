"""Main Safety Pipeline orchestrator implementing Defense-in-Depth architecture."""

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from PIL import Image

from config import settings
from config.settings import PipelineConfig
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

    Implements Defense-in-Depth architecture. Her katman PipelineConfig
    içindeki enable_* flag'leri ile bağımsız olarak açılıp kapatılabilir.

    Preset kullanımı:
        cfg = PipelineConfig.preset_full()
        pipeline = SafetyPipeline(pipeline_config=cfg)

    Manuel flag kullanımı:
        cfg = PipelineConfig(enable_clip=True, enable_vlm=False, ...)
        pipeline = SafetyPipeline(pipeline_config=cfg)
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        lazy_load: bool = True,
    ):
        self.cfg = pipeline_config or settings.pipeline
        self.lazy_load = lazy_load

        self.agent = SafetyAgent()
        self.diffusion = DiffusionGenerator()
        self.coca = CoCaEmbedder()
        self.clip = CLIPEmbedder()
        self.nudenet = NudeNetChecker()
        self.vlm = VLMChecker()

        self.metrics = BenchmarkMetrics()

        if not lazy_load:
            self._load_all()

    def _load_all(self) -> None:
        """Aktif bileşenleri belleğe yükle."""
        if self.cfg.enable_pre_check:
            self.agent.load()
        self.diffusion.load()
        if self.cfg.enable_clip:
            self.clip.load()
        if self.cfg.enable_coca:
            self.coca.load()
        if self.cfg.enable_nudenet:
            self.nudenet.load()
        if self.cfg.enable_vlm:
            self.vlm.load()

    def _unload_all(self) -> None:
        """Tüm modelleri bellekten boşalt."""
        self.agent.unload()
        self.diffusion.unload()
        self.coca.unload()
        self.clip.unload()
        self.nudenet.unload()
        self.vlm.unload()
        self._free_memory()

    def _free_memory(self) -> None:
        """Garbage collection ve CUDA cache temizleme."""
        if self.cfg.force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _unload_model(self, model) -> None:
        """unload_after_use etkinse modeli bellekten boşalt."""
        if self.cfg.unload_after_use:
            model.unload()
            self._free_memory()

    # ------------------------------------------------------------------
    # Post-check ortak mantık (generate ve check_image tarafından paylaşılır)
    # ------------------------------------------------------------------

    def _run_post_checks(
        self,
        image: Image.Image,
        result: PipelineResult,
    ) -> bool:
        """
        Aktif post-checker'ları çalıştır.
        Döner: post_safe (True = tüm kontroller geçti)
        """
        cfg = self.cfg
        post_safe = True

        # --- CLIP ---
        if cfg.enable_clip:
            t0 = time.perf_counter()
            clip_result = self.clip.check_safety(image)
            result.clip_result = clip_result
            result.timings["clip"] = (time.perf_counter() - t0) * 1000
            self._unload_model(self.clip)

            if clip_result.is_safe:
                result.stage_status["clip"] = "PASSED"
            else:
                triggered_str = ", ".join(clip_result.triggered_concepts[:3])
                result.stage_status["clip"] = f"CAUGHT ({triggered_str})"
                post_safe = False
        else:
            result.stage_status["clip"] = "SKIPPED"

        # --- CoCa captioning + Agent caption re-check ---
        if cfg.enable_coca:
            t0 = time.perf_counter()
            coca_result = self.coca.analyze(image)
            result.coca_result = coca_result
            result.timings["coca"] = (time.perf_counter() - t0) * 1000
            result.stage_status["coca"] = "COMPLETED"
            self._unload_model(self.coca)

            if cfg.enable_pre_check:
                t0 = time.perf_counter()
                caption_check = self.agent.check_caption(coca_result.caption)
                result.caption_recheck = caption_check
                result.timings["caption_recheck"] = (time.perf_counter() - t0) * 1000
                self._unload_model(self.agent)

                if caption_check.is_safe:
                    result.stage_status["caption_recheck"] = "PASSED"
                else:
                    result.stage_status["caption_recheck"] = (
                        f"CAUGHT ({caption_check.category or 'unsafe'})"
                    )
                    post_safe = False
            else:
                result.stage_status["caption_recheck"] = "SKIPPED"
        else:
            result.stage_status["coca"] = "SKIPPED"
            result.stage_status["caption_recheck"] = "SKIPPED"

        # --- NudeNet ---
        if cfg.enable_nudenet:
            t0 = time.perf_counter()
            nudenet_result = self.nudenet.check(image)
            result.nudenet_result = nudenet_result
            result.timings["nudenet"] = (time.perf_counter() - t0) * 1000

            if nudenet_result.is_nsfw:
                classes = ", ".join(nudenet_result.triggered_classes[:2])
                result.stage_status["nudenet"] = f"CAUGHT ({classes})"
                post_safe = False
            else:
                result.stage_status["nudenet"] = "PASSED"
        else:
            result.stage_status["nudenet"] = "SKIPPED"

        # --- VLM ---
        if cfg.enable_vlm:
            t0 = time.perf_counter()
            vlm_result = self.vlm.analyze(image)
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

        return post_safe

    def _finalize_result(
        self,
        result: PipelineResult,
        post_safe: bool,
        save_output: bool,
        output_dir: Path | None,
        image_index: int | None,
        start_time: float,
    ) -> PipelineResult:
        """Karar ver, görseli kaydet, metrikleri kaydet."""
        if post_safe:
            result.decision = SafetyDecision.SAFE
            result.stage_status["final"] = "SAFE"
        else:
            result.decision = SafetyDecision.UNSAFE_BLURRED
            result.stage_status["final"] = "UNSAFE_BLURRED"

        if save_output and result.image is not None:
            if image_index is not None:
                base_name = f"{image_index:04d}.png"
            else:
                from datetime import datetime
                base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            save_dir = output_dir or settings.paths.output_dir

            if result.decision == SafetyDecision.SAFE:
                result.output_path = save_image(
                    result.image, filename=base_name,
                    output_dir=save_dir, subfolder="safe",
                )
                result.original_path = result.output_path
            else:
                blurred_image = apply_blur(result.image)
                result.original_path, result.blurred_path = save_image_pair(
                    original=result.image,
                    blurred=blurred_image,
                    base_name=base_name,
                    output_dir=save_dir,
                )
                result.output_path = result.blurred_path
                result.image = blurred_image

        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        self._record_metrics(result)
        return result

    # ------------------------------------------------------------------
    # Genel amaçlı run(): prompt → generate → post-check
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        seed: int | None = None,
        guidance_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        save_output: bool = True,
        output_dir: Path | None = None,
        image_index: int | None = None,
    ) -> PipelineResult:
        """Prompt al, görsel üret, tüm aktif güvenlik katmanlarından geçir."""
        start_time = time.perf_counter()
        cfg = self.cfg

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
                result.stage_status["pre_check"] = (
                    f"BLOCKED ({pre_check.category or 'unsafe'})"
                )
                result.decision = SafetyDecision.BLOCKED
                result.total_time_ms = (time.perf_counter() - start_time) * 1000
                self._record_metrics(result)
                return result
        else:
            result.stage_status["pre_check"] = "SKIPPED"

        # === Stage 2: Image generation ===
        t0 = time.perf_counter()
        gen_result = self.diffusion.generate(
            prompt=prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        result.generation = gen_result
        result.image = gen_result.image
        result.timings["generation"] = (time.perf_counter() - t0) * 1000
        result.stage_status["generation"] = "COMPLETED"
        self._unload_model(self.diffusion)

        # === Stage 3: Post-generation checks ===
        post_safe = self._run_post_checks(gen_result.image, result)

        return self._finalize_result(
            result, post_safe, save_output, output_dir, image_index, start_time
        )

    # ------------------------------------------------------------------
    # check_image(): Hazır görsel → post-check (generate yok)
    # ------------------------------------------------------------------

    def check_image(
        self,
        image: Image.Image,
        source_path: Path | None = None,
        save_output: bool = True,
        output_dir: Path | None = None,
        image_index: int | None = None,
    ) -> PipelineResult:
        """Var olan bir görseli generate olmadan post-checker'lardan geçir.

        pre_check ve generation aşamaları her zaman atlanır.
        Hangi post-checker'ların çalışacağı PipelineConfig enable_* flag'leri
        ile belirlenir.
        """
        start_time = time.perf_counter()
        label = str(source_path) if source_path else "external_image"
        result = PipelineResult(prompt=label)

        result.stage_status["pre_check"] = "SKIPPED (image mode)"
        result.stage_status["generation"] = "SKIPPED (image mode)"
        result.image = image

        post_safe = self._run_post_checks(image, result)

        return self._finalize_result(
            result, post_safe, save_output, output_dir, image_index, start_time
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def run_batch(
        self,
        prompts: list[dict],
        save_output: bool = True,
        output_dir: Path | None = None,
    ) -> list[PipelineResult]:
        """Birden fazla prompt için pipeline çalıştır."""
        results = []
        for idx, row in enumerate(prompts):
            result = self.run(
                prompt=row["prompt"],
                seed=row.get("sd_seed"),
                guidance_scale=row.get("sd_guidance_scale"),
                width=row.get("sd_image_width"),
                height=row.get("sd_image_height"),
                save_output=save_output,
                output_dir=output_dir,
                image_index=idx + 1 if output_dir else None,
            )
            results.append(result)
        return results

    def check_image_batch(
        self,
        image_paths: list[Path],
        save_output: bool = True,
        output_dir: Path | None = None,
    ) -> list[PipelineResult]:
        """Birden fazla görsel dosyası için check_image çalıştır."""
        results = []
        for idx, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")
            result = self.check_image(
                image=image,
                source_path=img_path,
                save_output=save_output,
                output_dir=output_dir,
                image_index=idx + 1 if output_dir else None,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _record_metrics(self, result: PipelineResult) -> None:
        if not settings.benchmark_mode:
            return
        for layer_name, latency in result.timings.items():
            self.metrics.record_layer_time(layer_name, latency)
        self.metrics.record_pipeline_run(result.total_time_ms)

    def get_metrics_summary(self) -> dict:
        return self.metrics.get_summary()

    def reset_metrics(self) -> None:
        self.metrics.reset()
