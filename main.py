"""CLI entry point for NSFW Safety Filtering Pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import settings
from pipeline import SafetyPipeline, PipelineResult, SafetyDecision

logging.basicConfig(level=logging.WARNING)

app = typer.Typer(name="safety-sd", help="NSFW Safety Filtering Pipeline")
console = Console()


# =============================================================================
# Excel Report Generation
# =============================================================================

def create_benchmark_report(
    results: list[PipelineResult],
    metrics: dict,
    output_path: Path,
) -> Path:
    """Create Excel report with detailed results and metrics summary."""
    
    # Sheet 1: Detailed Results
    detail_rows = []
    for r in results:
        detail_rows.append({
            "prompt": r.prompt,
            "pre_check_safe": r.pre_check.is_safe if r.pre_check else None,
            "pre_check_reason": r.pre_check.reason if r.pre_check else None,
            "image_path": str(r.output_path) if r.output_path else None,
            "decision": r.decision.value,
            "time_ms": round(r.total_time_ms, 1),
            "coca_caption": r.coca_result.caption if r.coca_result else None,
            "vlm_safe": r.vlm_result.is_safe if r.vlm_result else None,
            "vlm_reason": r.vlm_result.reason if r.vlm_result else None,
            "nudenet_safe": not r.nudenet_result.is_nsfw if r.nudenet_result else None,
        })
    
    df_details = pd.DataFrame(detail_rows)
    
    # Sheet 2: Summary Metrics
    safe_count = sum(1 for r in results if r.decision == SafetyDecision.SAFE)
    blocked_count = sum(1 for r in results if r.decision == SafetyDecision.BLOCKED)
    blurred_count = sum(1 for r in results if r.decision == SafetyDecision.UNSAFE_BLURRED)
    
    pre_check_unsafe = sum(1 for r in results if r.pre_check and not r.pre_check.is_safe)
    nudenet_unsafe = sum(1 for r in results if r.nudenet_result and r.nudenet_result.is_nsfw)
    vlm_unsafe = sum(1 for r in results if r.vlm_result and not r.vlm_result.is_safe)
    coca_unsafe = sum(1 for r in results if r.caption_recheck and not r.caption_recheck.is_safe)
    
    summary_data = {
        "Metric": [
            "Total Images",
            "Total Time (ms)",
            "Mean Latency (ms)",
            "Throughput (img/s)",
            "",
            "Final Safe",
            "Final Blocked",
            "Final Blurred",
            "",
            "Pre-check Unsafe",
            "NudeNet Unsafe",
            "VLM Unsafe", 
            "CoCa Re-check Unsafe",
        ],
        "Value": [
            metrics["total_images"],
            round(metrics["total_time_ms"], 1),
            round(metrics["mean_latency_ms"], 1),
            round(metrics["throughput_ips"], 3),
            "",
            safe_count,
            blocked_count,
            blurred_count,
            "",
            pre_check_unsafe,
            nudenet_unsafe,
            vlm_unsafe,
            coca_unsafe,
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    # Sheet 3: Layer Breakdown
    layer_rows = []
    for name, m in metrics.get("layers", {}).items():
        layer_rows.append({
            "layer": name,
            "mean_ms": round(m["mean_ms"], 1),
            "min_ms": round(m["min_ms"], 1),
            "max_ms": round(m["max_ms"], 1),
            "count": m["count"],
        })
    df_layers = pd.DataFrame(layer_rows) if layer_rows else pd.DataFrame()
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_details.to_excel(writer, sheet_name="Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_layers.empty:
            df_layers.to_excel(writer, sheet_name="Layer_Breakdown", index=False)
    
    return output_path


def load_prompts_from_excel(file_path: Path) -> list[str]:
    """Load prompts from Excel, auto-detecting 'prompt' column."""
    df = pd.read_excel(file_path)
    
    prompt_col = next((c for c in df.columns if "prompt" in c.lower()), None)
    if not prompt_col:
        raise ValueError(f"No 'prompt' column found. Columns: {list(df.columns)}")
    
    return df[prompt_col].dropna().astype(str).tolist()


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
    input_file: Optional[Path] = typer.Option(None, "-i", "--input-file", help="Excel with prompts"),
    num_samples: Optional[int] = typer.Option(None, "-n", "--samples"),
    mode: str = typer.Option("full", "-m", "--mode"),
    output_report: Optional[Path] = typer.Option(None, "-o", "--output", help="Output Excel report"),
    save_images: bool = typer.Option(False, "--save-images", help="Save generated images"),
):
    """Run benchmark and generate Excel report."""
    
    # Load prompts
    if input_file:
        if not input_file.exists():
            console.print(f"[red]File not found: {input_file}[/red]")
            raise typer.Exit(1)
        prompts = load_prompts_from_excel(input_file)
        console.print(f"[green]Loaded {len(prompts)} prompts[/green]")
    else:
        prompts = [
            "A sunset over the ocean",
            "A cat playing with yarn",
            "Portrait in a garden",
            "Abstract geometric art",
            "Mountain landscape with snow",
        ]
    
    # Apply sample limit
    if num_samples:
        prompts = prompts[:num_samples]
    
    if not prompts:
        console.print("[red]No prompts to process[/red]")
        raise typer.Exit(1)
    
    settings.benchmark_mode = True
    console.print(f"[bold]Running {len(prompts)} prompts in '{mode}' mode...[/bold]\n")
    
    # Run pipeline
    pipeline = SafetyPipeline(mode=mode)
    
    with console.status("[green]Generating..."):
        results = pipeline.run_batch(prompts=prompts, save_output=save_images)
    
    metrics = pipeline.get_metrics_summary()
    
    # Display results table
    _print_results_table(results)
    _print_metrics(metrics, results)
    
    # Generate Excel report
    if output_report is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_report = Path(f"benchmark_report_{timestamp}.xlsx")
    
    report_path = create_benchmark_report(results, metrics, output_report)
    console.print(f"\n[green]Report saved: {report_path}[/green]")


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


def _print_results_table(results: list[PipelineResult]) -> None:
    """Print results table."""
    table = Table(title="Results")
    table.add_column("#", width=4)
    table.add_column("Prompt", max_width=40)
    table.add_column("Decision")
    table.add_column("Time", justify="right")
    
    colors = {"safe": "green", "unsafe_blurred": "yellow", "blocked": "red"}
    
    for i, r in enumerate(results, 1):
        c = colors[r.decision.value]
        prompt = r.prompt[:37] + "..." if len(r.prompt) > 40 else r.prompt
        table.add_row(str(i), prompt, f"[{c}]{r.decision.value}[/{c}]", f"{r.total_time_ms:.0f}ms")
    
    console.print(table)


def _print_metrics(metrics: dict, results: list[PipelineResult]) -> None:
    """Print metrics summary."""
    safe = sum(1 for r in results if r.decision == SafetyDecision.SAFE)
    blocked = sum(1 for r in results if r.decision == SafetyDecision.BLOCKED)
    blurred = sum(1 for r in results if r.decision == SafetyDecision.UNSAFE_BLURRED)
    
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
