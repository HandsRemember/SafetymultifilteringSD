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
from pipeline import SafetyPipeline, PipelineResult, SafetyDecision

logging.basicConfig(level=logging.WARNING)

app = typer.Typer(name="safety-sd", help="NSFW Safety Filtering Pipeline")
console = Console()

PROGRESS_FILENAME = "progress.jsonl"


# =============================================================================
# Input Loading (Excel + CSV)
# =============================================================================

def load_prompts_from_file(file_path: Path) -> list[str]:
    """Load prompts from Excel (.xlsx/.xls) or CSV file."""
    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx, .xls, or .csv")

    prompt_col = next((c for c in df.columns if "prompt" in c.lower()), None)
    if not prompt_col:
        raise ValueError(f"No 'prompt' column found. Columns: {list(df.columns)}")

    return df[prompt_col].dropna().astype(str).tolist()


# =============================================================================
# Benchmark Output Directory
# =============================================================================

def create_benchmark_dir() -> Path:
    """Create timestamped benchmark output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = Path("outputs") / f"benchmark_results_{timestamp}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir


# =============================================================================
# Progress Checkpoint (JSONL)
# =============================================================================

def _result_to_row(result: PipelineResult, index: int) -> dict:
    """Convert PipelineResult to a serializable dict for JSONL."""
    nudenet_regions = ""
    if result.nudenet_result and result.nudenet_result.triggered_classes:
        nudenet_regions = ", ".join(result.nudenet_result.triggered_classes)

    return {
        "index": index,
        "prompt": result.prompt,
        "pre_check_safe": result.pre_check.is_safe if result.pre_check else None,
        "pre_check_reason": result.pre_check.reason if result.pre_check else None,
        "image_path": str(result.output_path) if result.output_path else None,
        "decision": result.decision.value if result.decision else None,
        "time_ms": round(result.total_time_ms, 1),
        "coca_caption": result.coca_result.caption if result.coca_result else None,
        "vlm_safe": result.vlm_result.is_safe if result.vlm_result else None,
        "vlm_reason": result.vlm_result.reason if result.vlm_result else None,
        "nudenet_safe": (not result.nudenet_result.is_nsfw) if result.nudenet_result else None,
        "nudenet_exposed_regions": nudenet_regions,
        "timings": {k: round(v, 1) for k, v in result.timings.items()},
    }


def append_progress(progress_file: Path, row: dict) -> None:
    """Append one result row to progress JSONL file."""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress(progress_file: Path) -> list[dict]:
    """Load all completed rows from progress JSONL file."""
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
# Excel Report Generation (from row dicts)
# =============================================================================

def _build_metrics_from_rows(rows: list[dict]) -> dict:
    """Reconstruct benchmark metrics from JSONL rows."""
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
    """Create Excel report from row dicts (JSONL-compatible)."""
    metrics = _build_metrics_from_rows(rows)

    # Sheet 1: Detailed Results (timings kolonu haric)
    detail_cols = [
        "index", "prompt", "pre_check_safe", "pre_check_reason",
        "image_path", "decision", "time_ms", "coca_caption",
        "vlm_safe", "vlm_reason", "nudenet_safe", "nudenet_exposed_regions",
    ]
    df_details = pd.DataFrame([{k: r.get(k) for k in detail_cols} for r in rows])

    # Sheet 2: Summary
    safe_count = sum(1 for r in rows if r["decision"] == "safe")
    blocked_count = sum(1 for r in rows if r["decision"] == "blocked")
    blurred_count = sum(1 for r in rows if r["decision"] == "unsafe_blurred")

    pre_unsafe = sum(1 for r in rows if r.get("pre_check_safe") is False)
    nn_unsafe = sum(1 for r in rows if r.get("nudenet_safe") is False)
    vlm_unsafe = sum(1 for r in rows if r.get("vlm_safe") is False)
    coca_unsafe = sum(1 for r in rows if r.get("coca_caption") and r.get("pre_check_safe") is False)

    summary_data = {
        "Metric": [
            "Total Images", "Total Time (ms)", "Mean Latency (ms)",
            "Throughput (img/s)", "",
            "Final Safe", "Final Blocked", "Final Blurred", "",
            "Pre-check Unsafe", "NudeNet Unsafe", "VLM Unsafe",
        ],
        "Value": [
            metrics["total_images"], metrics["total_time_ms"],
            metrics["mean_latency_ms"], metrics["throughput_ips"], "",
            safe_count, blocked_count, blurred_count, "",
            pre_unsafe, nn_unsafe, vlm_unsafe,
        ],
    }
    df_summary = pd.DataFrame(summary_data)

    # Sheet 3: Layer Breakdown
    layer_rows = [
        {"layer": n, **v} for n, v in metrics["layers"].items()
    ]
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
    mode: str = typer.Option("full", "-m", "--mode"),
    no_save: bool = typer.Option(False, "--no-save"),
):
    """Generate single image with safety filtering."""
    console.print(Panel(f"[blue]{prompt}[/blue]", title="Prompt"))

    with console.status("[green]Processing..."):
        pipeline = SafetyPipeline(mode=mode)
        result = pipeline.run(prompt=prompt, seed=seed, save_output=not no_save)

    _print_result(result)


@app.command()
def benchmark(
    input_file: Optional[Path] = typer.Option(None, "-i", "--input-file", help="Excel (.xlsx) or CSV (.csv) with prompts"),
    num_samples: Optional[int] = typer.Option(None, "-n", "--samples"),
    mode: str = typer.Option("full", "-m", "--mode"),
    output_report: Optional[Path] = typer.Option(None, "-o", "--output", help="Output Excel report path"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from existing benchmark dir"),
):
    """Run benchmark with checkpoint support. Use --resume to continue."""

    # ---- Resume or fresh start ----
    if resume:
        benchmark_dir = Path(resume)
        if not benchmark_dir.is_dir():
            console.print(f"[red]Directory not found: {benchmark_dir}[/red]")
            raise typer.Exit(1)

        progress_file = benchmark_dir / PROGRESS_FILENAME
        completed_rows = load_progress(progress_file)
        completed_count = len(completed_rows)

        if not input_file:
            console.print("[red]--resume requires --input-file to know the full prompt list[/red]")
            raise typer.Exit(1)

        all_prompts = load_prompts_from_file(input_file)
        if num_samples:
            all_prompts = all_prompts[:num_samples]

        if completed_count >= len(all_prompts):
            console.print(f"[green]Already completed all {completed_count} prompts![/green]")
            _finalize_report(completed_rows, benchmark_dir, output_report)
            return

        remaining_prompts = all_prompts[completed_count:]
        start_index = completed_count + 1

        console.print(f"[yellow]Resuming from #{start_index} ({completed_count}/{len(all_prompts)} done)[/yellow]")
        console.print(f"[dim]Images → {benchmark_dir}[/dim]\n")

    else:
        # Fresh start
        if input_file:
            if not input_file.exists():
                console.print(f"[red]File not found: {input_file}[/red]")
                raise typer.Exit(1)
            all_prompts = load_prompts_from_file(input_file)
            console.print(f"[green]Loaded {len(all_prompts)} prompts from {input_file.name}[/green]")
        else:
            all_prompts = [
                "A sunset over the ocean",
                "A cat playing with yarn",
                "Portrait in a garden",
                "Abstract geometric art",
                "Mountain landscape with snow",
            ]

        if num_samples:
            all_prompts = all_prompts[:num_samples]

        if not all_prompts:
            console.print("[red]No prompts to process[/red]")
            raise typer.Exit(1)

        benchmark_dir = create_benchmark_dir()
        remaining_prompts = all_prompts
        start_index = 1
        completed_rows = []

        console.print(f"[bold]Running {len(all_prompts)} prompts in '{mode}' mode[/bold]")
        console.print(f"[dim]Images → {benchmark_dir}[/dim]\n")

    # ---- Run pipeline with checkpoint ----
    settings.benchmark_mode = True
    progress_file = benchmark_dir / PROGRESS_FILENAME
    pipeline = SafetyPipeline(mode=mode)
    total_prompts = len(all_prompts)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmark", total=total_prompts, completed=start_index - 1)

        for i, prompt in enumerate(remaining_prompts):
            current_index = start_index + i

            result = pipeline.run(
                prompt=prompt,
                save_output=True,
                output_dir=benchmark_dir,
                image_index=current_index,
            )

            row = _result_to_row(result, current_index)
            append_progress(progress_file, row)
            completed_rows.append(row)

            progress.update(task, advance=1)

    # ---- Finalize ----
    all_rows = load_progress(progress_file)
    _print_rows_table(all_rows)
    _print_rows_metrics(all_rows)
    _finalize_report(all_rows, benchmark_dir, output_report)


def _finalize_report(rows: list[dict], benchmark_dir: Path, output_report: Optional[Path]) -> None:
    """Generate final Excel report from all rows."""
    if output_report is None:
        output_report = benchmark_dir / "benchmark_report.xlsx"

    report_path = create_benchmark_report(rows, output_report)
    console.print(f"\n[green]Report saved: {report_path}[/green]")
    console.print(f"[green]Images dir: {benchmark_dir}[/green]")
    console.print(f"[green]Total: {len(rows)} prompts processed[/green]")


@app.command()
def check_prompt(prompt: str = typer.Argument(...)):
    """Check prompt safety without generating image."""
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
    """Print detailed result with stage-by-stage status."""
    
    # Stage status table
    stage_table = Table(title="Pipeline Stages", show_header=True)
    stage_table.add_column("Stage", style="cyan")
    stage_table.add_column("Status")
    stage_table.add_column("Time", justify="right")
    
    stage_order = ["pre_check", "generation", "coca", "caption_recheck", "nudenet", "vlm", "clip"]
    
    for stage in stage_order:
        if stage in r.stage_status:
            status = r.stage_status[stage]
            time_ms = r.timings.get(stage, 0)
            
            # Color based on status
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
    
    # Final result panel
    color = {"safe": "green", "unsafe_blurred": "yellow", "blocked": "red"}[r.decision.value]
    
    lines = [
        f"Decision: [{color}]{r.decision.value.upper()}[/{color}]",
        f"Total Time: {r.total_time_ms:.1f}ms",
    ]
    
    if r.coca_result:
        lines.append(f"Caption: {r.coca_result.caption}")
    
    if r.decision.value == "safe":
        if r.output_path:
            lines.append(f"Output: {r.output_path}")
    else:
        if r.original_path:
            lines.append(f"Original: {r.original_path}")
        if r.blurred_path:
            lines.append(f"Blurred: {r.blurred_path}")
    
    console.print(Panel("\n".join(lines), title="Final Result"))


def _print_rows_table(rows: list[dict]) -> None:
    """Print results table from row dicts."""
    table = Table(title="Results")
    table.add_column("#", width=4)
    table.add_column("Prompt", max_width=40)
    table.add_column("Decision")
    table.add_column("Time", justify="right")

    colors = {"safe": "green", "unsafe_blurred": "yellow", "blocked": "red"}

    for r in rows:
        c = colors.get(r["decision"], "white")
        prompt = r["prompt"][:37] + "..." if len(r["prompt"]) > 40 else r["prompt"]
        table.add_row(str(r["index"]), prompt, f"[{c}]{r['decision']}[/{c}]", f"{r['time_ms']:.0f}ms")

    console.print(table)


def _print_rows_metrics(rows: list[dict]) -> None:
    """Print metrics summary from row dicts."""
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
