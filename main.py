"""CLI entry point for NSFW Safety Filtering Pipeline."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from config import settings
from config.settings import PipelineConfig
from pipeline import SafetyPipeline, PipelineResult, SafetyDecision

logging.basicConfig(level=logging.WARNING)

app = typer.Typer(name="safety-sd", help="NSFW Safety Filtering Pipeline")
console = Console()

PROGRESS_FILENAME = "progress.jsonl"

# ---------------------------------------------------------------------------
# Pipeline preset mapping  (--mode flag → PipelineConfig)
# ---------------------------------------------------------------------------

PRESETS: dict[str, type] = {
    "full":      PipelineConfig.preset_full,
    "baseline":  PipelineConfig.preset_baseline,
    "clip_only": PipelineConfig.preset_clip_only,
}

_VALID_MODES = ", ".join(PRESETS.keys())


def _resolve_pipeline(mode: str | None) -> SafetyPipeline:
    """Resolve --mode string to a PipelineConfig preset and create SafetyPipeline.

    If mode is None, settings.pipeline is used directly so that enable_*
    flags in settings.py take effect without being overridden by a preset.
    """
    if mode is None:
        return SafetyPipeline(pipeline_config=settings.pipeline)

    if mode not in PRESETS:
        console.print(
            f"[red]Invalid mode: '{mode}'. Valid modes: {_VALID_MODES}[/red]"
        )
        raise typer.Exit(1)
    cfg = PRESETS[mode]()
    return SafetyPipeline(pipeline_config=cfg)


# =============================================================================
# Input Loading (Excel + CSV)
# =============================================================================

_EXTRA_COLUMNS = {
    "sd_seed": int,
    "sd_guidance_scale": float,
    "sd_image_width": int,
    "sd_image_height": int,
}


def load_prompts_from_file(file_path: Path) -> list[dict]:
    """Load prompt list from Excel or CSV file."""
    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .xlsx, .xls or .csv.")

    prompt_col = next((c for c in df.columns if "prompt" in c.lower()), None)
    if not prompt_col:
        raise ValueError(f"No 'prompt' column found. Available columns: {list(df.columns)}")

    col_map: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for key in _EXTRA_COLUMNS:
        if key.lower() in lower_cols:
            col_map[key] = lower_cols[key.lower()]

    rows: list[dict] = []
    for _, df_row in df.iterrows():
        prompt_val = df_row[prompt_col]
        if pd.isna(prompt_val):
            continue

        entry: dict = {"prompt": str(prompt_val)}
        for key, cast in _EXTRA_COLUMNS.items():
            if key in col_map:
                val = df_row[col_map[key]]
                entry[key] = cast(val) if pd.notna(val) else None
            else:
                entry[key] = None
        rows.append(entry)

    return rows


# =============================================================================
# Benchmark Output Directory
# =============================================================================

def create_benchmark_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = Path("outputs") / f"benchmark_results_{timestamp}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir


# =============================================================================
# Progress Checkpoint (JSONL)
# =============================================================================

def _result_to_row(result: PipelineResult, index: int) -> dict:
    """PipelineResult'ı JSONL için serileştirilebilir dict'e çevir."""
    nudenet_regions = ""
    if result.nudenet_result and result.nudenet_result.triggered_classes:
        nudenet_regions = ", ".join(result.nudenet_result.triggered_classes)

    clip_triggered = ""
    if result.clip_result and result.clip_result.triggered_concepts:
        clip_triggered = ", ".join(result.clip_result.triggered_concepts)

    gen = result.generation
    return {
        "index": index,
        "prompt": result.prompt,
        "sd_seed": gen.seed if gen else None,
        "sd_guidance_scale": gen.guidance_scale if gen else None,
        "sd_width": gen.width if gen else None,
        "sd_height": gen.height if gen else None,
        "pre_check_safe": result.pre_check.is_safe if result.pre_check else None,
        "pre_check_reason": result.pre_check.reason if result.pre_check else None,
        "image_path": str(result.output_path) if result.output_path else None,
        "decision": result.decision.value if result.decision else None,
        "time_ms": round(result.total_time_ms, 1),
        "coca_caption": result.coca_result.caption if result.coca_result else None,
        "clip_safe": (result.clip_result.is_safe) if result.clip_result else None,
        "clip_triggered": clip_triggered if clip_triggered else None,
        "vlm_safe": result.vlm_result.is_safe if result.vlm_result else None,
        "vlm_reason": result.vlm_result.reason if result.vlm_result else None,
        "nudenet_safe": (not result.nudenet_result.is_nsfw) if result.nudenet_result else None,
        "nudenet_exposed_regions": nudenet_regions,
        "timings": {k: round(v, 1) for k, v in result.timings.items()},
    }


def append_progress(progress_file: Path, row: dict) -> None:
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress(progress_file: Path) -> list[dict]:
    if not progress_file.exists():
        return []
    rows = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# =============================================================================
# Excel Report Generation
# =============================================================================

def _build_metrics_from_rows(rows: list[dict]) -> dict:
    total_time = sum(r["time_ms"] for r in rows)
    total_images = len(rows)

    layer_data: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for layer_name, latency in r.get("timings", {}).items():
            layer_data[layer_name].append(latency)

    layers = {}
    for name, latencies in layer_data.items():
        layers[name] = {
            "mean_ms": round(sum(latencies) / len(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "count": len(latencies),
        }

    return {
        "total_images": total_images,
        "total_time_ms": round(total_time, 1),
        "mean_latency_ms": round(total_time / total_images, 1) if total_images else 0.0,
        "throughput_ips": round(total_images / (total_time / 1000), 3) if total_time else 0.0,
        "layers": layers,
    }


def create_benchmark_report(rows: list[dict], output_path: Path) -> Path:
    metrics = _build_metrics_from_rows(rows)

    detail_cols = [
        "index", "prompt", "sd_seed", "sd_guidance_scale", "sd_width", "sd_height",
        "pre_check_safe", "pre_check_reason",
        "image_path", "decision", "time_ms", "coca_caption",
        "clip_safe", "clip_triggered",
        "vlm_safe", "vlm_reason", "nudenet_safe", "nudenet_exposed_regions",
    ]
    df_details = pd.DataFrame([{k: r.get(k) for k in detail_cols} for r in rows])

    safe_count = sum(1 for r in rows if r["decision"] == "safe")
    blocked_count = sum(1 for r in rows if r["decision"] == "blocked")
    blurred_count = sum(1 for r in rows if r["decision"] == "unsafe_blurred")

    pre_unsafe = sum(1 for r in rows if r.get("pre_check_safe") is False)
    nn_unsafe = sum(1 for r in rows if r.get("nudenet_safe") is False)
    vlm_unsafe = sum(1 for r in rows if r.get("vlm_safe") is False)
    clip_unsafe = sum(1 for r in rows if r.get("clip_safe") is False)

    summary_data = {
        "Metric": [
            "Total Images", "Total Time (ms)", "Mean Latency (ms)",
            "Throughput (img/s)", "",
            "Final Safe", "Final Blocked", "Final Blurred", "",
            "Pre-check Unsafe", "CLIP Unsafe", "NudeNet Unsafe", "VLM Unsafe",
        ],
        "Value": [
            metrics["total_images"], metrics["total_time_ms"],
            metrics["mean_latency_ms"], metrics["throughput_ips"], "",
            safe_count, blocked_count, blurred_count, "",
            pre_unsafe, clip_unsafe, nn_unsafe, vlm_unsafe,
        ],
    }
    df_summary = pd.DataFrame(summary_data)

    layer_rows = [{"layer": n, **v} for n, v in metrics["layers"].items()]
    df_layers = pd.DataFrame(layer_rows) if layer_rows else pd.DataFrame()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_details.to_excel(writer, sheet_name="Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_layers.empty:
            df_layers.to_excel(writer, sheet_name="Layer_Breakdown", index=False)

    return output_path


# =============================================================================
# CLI Commands
# =============================================================================

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt"),
    seed: Optional[int] = typer.Option(None, "-s", "--seed"),
    width: Optional[int] = typer.Option(None, "-W", "--width"),
    height: Optional[int] = typer.Option(None, "-H", "--height"),
    scale: Optional[float] = typer.Option(None, "--scale", help="Guidance scale / CFG"),
    mode: Optional[str] = typer.Option(None, "-m", "--mode", help=f"Preset: {_VALID_MODES}. If omitted, settings.py enable_* flags are used directly."),
    no_save: bool = typer.Option(False, "--no-save"),
):
    """Generate a single image from prompt and run safety filters."""
    console.print(Panel(f"[blue]{prompt}[/blue]", title="Prompt"))

    with console.status("[green]Processing..."):
        pipeline = _resolve_pipeline(mode)
        result = pipeline.run(
            prompt=prompt,
            seed=seed,
            guidance_scale=scale,
            width=width,
            height=height,
            save_output=not no_save,
        )

    _print_result(result)


@app.command()
def check(
    path: Path = typer.Argument(..., help="Image file or directory"),
    mode: Optional[str] = typer.Option(None, "-m", "--mode", help=f"Preset: {_VALID_MODES}. If omitted, settings.py enable_* flags are used directly."),
    no_save: bool = typer.Option(False, "--no-save"),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output-dir"),
):
    """Run post-checkers on existing image(s) without generation.

    Accepts a single file or a directory. pre_check and generation are always
    skipped. Which checkers run is controlled by enable_* flags in settings.py
    (or the --mode preset).

    Examples:\n
      python main.py check outputs/image.png\n
      python main.py check outputs/safe/ --mode baseline\n
      python main.py check outputs/ --mode full --no-save
    """
    if not path.exists():
        console.print(f"[red]Not found: {path}[/red]")
        raise typer.Exit(1)

    if path.is_file():
        image_paths = [path]
    elif path.is_dir():
        image_paths = sorted(
            list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
        )
        if not image_paths:
            console.print(f"[yellow]No PNG/JPG images found in directory: {path}[/yellow]")
            raise typer.Exit(0)
    else:
        console.print(f"[red]Invalid path: {path}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(f"[cyan]{len(image_paths)} image(s)[/cyan] · mode: [bold]{mode or 'settings.py'}[/bold]", title="Check")
    )

    pipeline = _resolve_pipeline(mode)

    for img_path in image_paths:
        console.rule(f"[dim]{img_path.name}[/dim]")
        try:
            image = __import__("PIL.Image", fromlist=["Image"]).open(img_path).convert("RGB")
        except Exception as e:
            console.print(f"[red]Could not open image ({img_path.name}): {e}[/red]")
            continue

        result = pipeline.check_image(
            image=image,
            source_path=img_path,
            save_output=not no_save,
            output_dir=output_dir,
        )
        _print_result(result)


@app.command()
def benchmark(
    input_file: Optional[Path] = typer.Option(None, "-i", "--input-file", help="Excel (.xlsx) or CSV (.csv) with prompts"),
    source_dir: Optional[Path] = typer.Option(
        None, "--source-dir",
        help="Directory of existing images. When provided, generation is skipped and only post-checks run."
    ),
    num_samples: Optional[int] = typer.Option(None, "-n", "--samples"),
    mode: Optional[str] = typer.Option(None, "-m", "--mode", help=f"Preset: {_VALID_MODES}. If omitted, settings.py enable_* flags are used directly."),
    output_report: Optional[Path] = typer.Option(None, "-o", "--output"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from an existing benchmark directory"),
):
    """Run benchmark. Use --source-dir to apply post-checks to an existing image set.

    Examples:\n
      # Classic: prompt → generate → post-check\n
      python main.py benchmark -i prompts.xlsx --mode full\n\n
      # Existing images: post-check only\n
      python main.py benchmark --source-dir outputs/images/ --mode full\n
      python main.py benchmark --source-dir outputs/images/ -n 50 --mode baseline
    """
    use_source_dir = source_dir is not None

    # ---- Resume or fresh start ----
    if resume:
        benchmark_dir = Path(resume)
        if not benchmark_dir.is_dir():
            console.print(f"[red]Directory not found: {benchmark_dir}[/red]")
            raise typer.Exit(1)

        progress_file = benchmark_dir / PROGRESS_FILENAME
        completed_rows = load_progress(progress_file)
        completed_count = len(completed_rows)

        if use_source_dir:
            all_items = _collect_images(source_dir, num_samples)
        else:
            if not input_file:
                console.print("[red]--resume requires --input-file or --source-dir.[/red]")
                raise typer.Exit(1)
            all_items = load_prompts_from_file(input_file)
            if num_samples:
                all_items = all_items[:num_samples]

        if completed_count >= len(all_items):
            console.print(f"[green]All {completed_count} items already completed![/green]")
            _finalize_report(completed_rows, benchmark_dir, output_report)
            return

        remaining_items = all_items[completed_count:]
        start_index = completed_count + 1
        console.print(
            f"[yellow]Resuming from #{start_index} ({completed_count}/{len(all_items)} done)[/yellow]"
        )
        console.print(f"[dim]Output → {benchmark_dir}[/dim]\n")

    else:
        if use_source_dir:
            if not source_dir.is_dir():
                console.print(f"[red]Directory not found: {source_dir}[/red]")
                raise typer.Exit(1)
            all_items = _collect_images(source_dir, num_samples)
            console.print(
                f"[green]{len(all_items)} images found: {source_dir}[/green]"
            )
        elif input_file:
            if not input_file.exists():
                console.print(f"[red]File not found: {input_file}[/red]")
                raise typer.Exit(1)
            all_items = load_prompts_from_file(input_file)
            if num_samples:
                all_items = all_items[:num_samples]
            console.print(
                f"[green]{len(all_items)} prompts loaded from {input_file.name}[/green]"
            )
        else:
            all_items = [
                {"prompt": p} for p in [
                    "A sunset over the ocean",
                    "A cat playing with yarn",
                    "Portrait in a garden",
                    "Abstract geometric art",
                    "Mountain landscape with snow",
                ]
            ]
            if num_samples:
                all_items = all_items[:num_samples]

        if not all_items:
            console.print("[red]No items to process.[/red]")
            raise typer.Exit(1)

        benchmark_dir = create_benchmark_dir()
        remaining_items = all_items
        start_index = 1
        completed_rows = []

        mode_label = mode or "settings.py"
        label = "images (post-check)" if use_source_dir else f"prompts ('{mode_label}' mode)"
        console.print(f"[bold]Running {len(all_items)} {label}[/bold]")
        console.print(f"[dim]Output → {benchmark_dir}[/dim]\n")

    # ---- Pipeline oluştur ----
    settings.benchmark_mode = True
    progress_file = benchmark_dir / PROGRESS_FILENAME
    pipeline = _resolve_pipeline(mode)
    total_items = len(all_items)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmark", total=total_items, completed=start_index - 1)

        for i, item in enumerate(remaining_items):
            current_index = start_index + i

            if use_source_dir:
                try:
                    image = __import__("PIL.Image", fromlist=["Image"]).open(item).convert("RGB")
                except Exception as e:
                    console.print(f"[red]Could not open image ({item.name}): {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                result = pipeline.check_image(
                    image=image,
                    source_path=item,
                    save_output=True,
                    output_dir=benchmark_dir,
                    image_index=current_index,
                )
            else:
                result = pipeline.run(
                    prompt=item["prompt"],
                    seed=item.get("sd_seed"),
                    guidance_scale=item.get("sd_guidance_scale"),
                    width=item.get("sd_image_width"),
                    height=item.get("sd_image_height"),
                    save_output=True,
                    output_dir=benchmark_dir,
                    image_index=current_index,
                )

            row = _result_to_row(result, current_index)
            append_progress(progress_file, row)
            completed_rows.append(row)
            progress.update(task, advance=1)

    all_rows = load_progress(progress_file)
    _print_rows_table(all_rows)
    _print_rows_metrics(all_rows)
    _finalize_report(all_rows, benchmark_dir, output_report)


def _collect_images(source_dir: Path, num_samples: Optional[int]) -> list[Path]:
    """Collect all PNG/JPG files from a directory."""
    images = sorted(
        list(source_dir.glob("*.png"))
        + list(source_dir.glob("*.jpg"))
        + list(source_dir.glob("*.jpeg"))
    )
    if num_samples:
        images = images[:num_samples]
    return images


def _finalize_report(rows: list[dict], benchmark_dir: Path, output_report: Optional[Path]) -> None:
    if output_report is None:
        output_report = benchmark_dir / "benchmark_report.xlsx"
    report_path = create_benchmark_report(rows, output_report)
    console.print(f"\n[green]Report saved: {report_path}[/green]")
    console.print(f"[green]Images dir: {benchmark_dir}[/green]")
    console.print(f"[green]Total: {len(rows)} items processed[/green]")


@app.command()
def check_prompt(prompt: str = typer.Argument(...)):
    """Check prompt safety without generating an image."""
    from models import SafetyAgent

    agent = SafetyAgent()
    result = agent.check_prompt(prompt)

    status = "[green]SAFE[/green]" if result.is_safe else "[red]UNSAFE[/red]"
    console.print(Panel(
        f"Status: {status}\nReason: {result.reason}\nCategory: {result.category or 'N/A'}",
        title="Safety Check"
    ))


# =============================================================================
# Display Helpers
# =============================================================================

def _print_result(r: PipelineResult) -> None:
    stage_table = Table(title="Pipeline Stages", show_header=True)
    stage_table.add_column("Stage", style="cyan")
    stage_table.add_column("Status")
    stage_table.add_column("Time", justify="right")

    stage_order = ["pre_check", "generation", "clip", "coca", "caption_recheck", "nudenet", "vlm"]

    for stage in stage_order:
        if stage in r.stage_status:
            status = r.stage_status[stage]
            time_ms = r.timings.get(stage, 0)

            if "PASSED" in status or "COMPLETED" in status:
                status_str = f"[green]{status}[/green]"
            elif "CAUGHT" in status or "BLOCKED" in status:
                status_str = f"[red]{status}[/red]"
            elif "SKIPPED" in status:
                status_str = f"[dim]{status}[/dim]"
            else:
                status_str = status

            time_str = f"{time_ms:.0f}ms" if time_ms > 0 else "-"
            stage_table.add_row(stage, status_str, time_str)

    console.print(stage_table)

    color = {"safe": "green", "unsafe_blurred": "yellow", "blocked": "red"}[r.decision.value]

    lines = [
        f"Decision: [{color}]{r.decision.value.upper()}[/{color}]",
        f"Total Time: {r.total_time_ms:.1f}ms",
    ]

    if r.coca_result:
        lines.append(f"Caption: {r.coca_result.caption}")

    if r.clip_result:
        if r.clip_result.triggered_concepts:
            lines.append(f"CLIP Triggered: {', '.join(r.clip_result.triggered_concepts)}")
        top_concept = f" ({r.clip_result.matched_concept})" if r.clip_result.matched_concept else ""
        lines.append(f"CLIP Max Similarity: {r.clip_result.max_similarity:.3f}{top_concept}")

    if r.decision.value == "safe":
        if r.output_path:
            lines.append(f"Output: {r.output_path}")
    else:
        if r.original_path:
            lines.append(f"Original: {r.original_path}")
        if r.blurred_path:
            lines.append(f"Blurred: {r.blurred_path}")

    console.print(Panel("\n".join(lines), title="Result"))


def _print_rows_table(rows: list[dict]) -> None:
    table = Table(title="Results")
    table.add_column("#", width=4)
    table.add_column("Prompt / Image", max_width=40)
    table.add_column("Decision")
    table.add_column("Time", justify="right")

    colors = {"safe": "green", "unsafe_blurred": "yellow", "blocked": "red"}

    for r in rows:
        c = colors.get(r["decision"], "white")
        label = r["prompt"][:37] + "..." if len(r["prompt"]) > 40 else r["prompt"]
        table.add_row(str(r["index"]), label, f"[{c}]{r['decision']}[/{c}]", f"{r['time_ms']:.0f}ms")

    console.print(table)


def _print_rows_metrics(rows: list[dict]) -> None:
    metrics = _build_metrics_from_rows(rows)

    safe = sum(1 for r in rows if r["decision"] == "safe")
    blocked = sum(1 for r in rows if r["decision"] == "blocked")
    blurred = sum(1 for r in rows if r["decision"] == "unsafe_blurred")

    table = Table(title="Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Images", str(metrics["total_images"]))
    table.add_row("Total Time", f"{metrics['total_time_ms']:.0f}ms")
    table.add_row("Mean Latency", f"{metrics['mean_latency_ms']:.0f}ms")
    table.add_row("Throughput", f"{metrics['throughput_ips']:.2f} img/s")
    table.add_row("", "")
    table.add_row("[green]Safe[/green]", str(safe))
    table.add_row("[red]Blocked[/red]", str(blocked))
    table.add_row("[yellow]Blurred[/yellow]", str(blurred))

    console.print(table)


if __name__ == "__main__":
    app()
