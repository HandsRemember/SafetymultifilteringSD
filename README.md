# NSFW Safety Filtering Pipeline for Stable Diffusion

A multi-layered safety filtering framework for text-to-image generation, implementing a **Defense-in-Depth** architecture. This project is part of a Master's thesis on **NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Defense-in-Depth Safety Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────────┐         ┌──────────────────┐           │
│  │  Input   │───►│  Pre-Generation │──Safe?──►│ Stable Diffusion │           │
│  │  Prompt  │    │  Agent (Qwen)   │         │      1.5          │           │
│  └──────────┘    └────────┬────────┘         └────────┬─────────┘           │
│                           │                           │                      │
│                      [BLOCKED]                        ▼                      │
│                      if unsafe              ┌──────────────────┐            │
│                                             │  Generated Image  │            │
│                                             └────────┬─────────┘            │
│                                                      │                      │
│                    ┌─────────────────────────────────┼───────────────┐      │
│                    │           Post-Generation Analysis              │      │
│                    │  ┌─────────────┐  ┌─────────┐  ┌─────────────┐ │      │
│                    │  │    CoCa     │  │ NudeNet │  │  Qwen2-VL   │ │      │
│                    │  │ Captioning  │  │  Check  │  │   7B-VLM    │ │      │
│                    │  └──────┬──────┘  └────┬────┘  └──────┬──────┘ │      │
│                    │         │              │              │        │      │
│                    │         └──────────────┼──────────────┘        │      │
│                    │                        │                       │      │
│                    └────────────────────────┼───────────────────────┘      │
│                                             │                              │
│                                        All Safe?                           │
│                                       /         \                          │
│                                     Yes          No                        │
│                                      │            │                        │
│                               ┌──────▼──────┐ ┌──▼───────────┐             │
│                               │ Safe Output │ │Blurred Output│             │
│                               └─────────────┘ └──────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Pre-Generation Safety**: LLM-based prompt analysis using Qwen2.5-7B-Instruct (4-bit quantized)
- **Image Generation**: Stable Diffusion (Deliberate v5) with native safety checker disabled
- **Post-Generation Analysis**:
  - **CoCa** (Contrastive Captioners) for image-to-text captioning + Agent re-check
  - **NudeNet v3** for CNN-based nudity detection with bounding boxes and exposed region reporting
  - **Qwen2-VL-7B** Vision Language Model for semantic safety analysis
- **Flexible Pipeline Modes**: Full mode vs CLIP baseline comparison
- **VRAM Optimized**: Sequential model loading with auto-unload for RTX 3070 (8GB VRAM)
- **Benchmark System**: Excel/CSV input, Excel reports, per-layer timing, checkpoint/resume support

## Requirements

- Python 3.10+
- CUDA 11.8+ compatible GPU
- ~8GB VRAM (RTX 3070 or equivalent)
- ~15GB disk space for model downloads

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HandsRemember/SafetymultifilteringSD.git
cd SafetymultifilteringSD
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. First Run (Model Download)

The first run will download required models (~15GB total):
- Deliberate v5 (~4GB) - Image generation
- Qwen2.5-7B-Instruct (~5GB) - Pre-generation safety agent (4-bit)
- CoCa ViT-L-14 (~1.5GB) - Image captioning
- Qwen2-VL-7B-Instruct (~5GB) - Vision-Language Model (4-bit)
- NudeNet v3 (~200MB) - Nudity detection
- CLIP ViT-L-14 (~1GB) - Baseline mode only

## Usage

### Basic Generation

```bash
python main.py generate "A beautiful sunset over the ocean"
```

### With Options

```bash
# Specify seed for reproducibility
python main.py generate "A cat playing with yarn" --seed 42

# Use baseline CLIP mode (faster, less accurate)
python main.py generate "A mountain landscape" --mode baseline

# Don't save output images
python main.py generate "A robot reading a book" --no-save
```

### Prompt Safety Check (No Generation)

```bash
python main.py check-prompt "Your prompt here"
```

### Benchmark Mode

```bash
# Default test prompts (5 built-in samples)
python main.py benchmark

# From Excel file (all rows)
python main.py benchmark -i prompts.xlsx

# From CSV file
python main.py benchmark -i prompts.csv

# First 20 samples only
python main.py benchmark -i prompts.xlsx -n 20

# Custom output report path
python main.py benchmark -i prompts.xlsx -o my_report.xlsx

# Use baseline CLIP mode for comparison
python main.py benchmark -i prompts.xlsx -m baseline
```

### Resume (Checkpoint Support)

Long benchmarks can be interrupted and resumed. Each prompt result is saved to `progress.jsonl` immediately after completion.

```bash
# Start a benchmark (creates outputs/benchmark_results_20260222_143015/)
python main.py benchmark -i dataset.xlsx -n 500

# If interrupted, resume from where it left off:
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 -i dataset.xlsx -n 500

# If already completed, just regenerates the Excel report:
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 -i dataset.xlsx
```

### Benchmark Output Structure

Each benchmark run creates a timestamped directory with images and reports:

```
outputs/benchmark_results_20260222_143015/
├── safe/
│   ├── 0001.png              # Safe images (original)
│   └── 0004.png
├── original/
│   ├── 0002.png              # Unsafe images (unblurred)
│   └── 0003.png
├── blurred/
│   ├── 0002.png              # Unsafe images (blurred)
│   └── 0003.png
├── progress.jsonl             # Checkpoint file (1 JSON line per prompt)
└── benchmark_report.xlsx      # Final Excel report (generated at end)
```

Images are named by their order in the input file (`0001.png`, `0002.png`, ...).

### Excel Report Output

Benchmark automatically generates an Excel report with 3 sheets:

**Sheet 1 - Results:**
| index | prompt | pre_check_safe | pre_check_reason | image_path | decision | time_ms | coca_caption | vlm_safe | vlm_reason | nudenet_safe | nudenet_exposed_regions |
|-------|--------|----------------|------------------|------------|----------|---------|--------------|----------|------------|--------------|------------------------|

**Sheet 2 - Summary:**
| Metric | Value |
|--------|-------|
| Total Images | 100 |
| Total Time (ms) | 234500 |
| Mean Latency (ms) | 2345 |
| Throughput (img/s) | 0.43 |
| Final Safe | 85 |
| Final Blocked | 10 |
| Final Blurred | 5 |
| Pre-check Unsafe | 10 |
| NudeNet Unsafe | 3 |
| VLM Unsafe | 2 |

**Sheet 3 - Layer Breakdown:**
| layer | mean_ms | min_ms | max_ms | count |
|-------|---------|--------|--------|-------|

### Input File Format

Supports `.xlsx`, `.xls`, and `.csv` files. Must have a column containing "prompt" (case-insensitive):
- `prompt`, `Prompt`, `prompt_tr`, `English_Prompt`, etc.

## Pipeline Modes

| Mode | Components | Use Case |
|------|------------|----------|
| `full` | Agent + SD + CoCa + NudeNet + VLM | Maximum safety, multi-layer defense (default) |
| `baseline` | Agent + SD + CLIP | Faster processing, CLIP-based safety check |
| `clip_only` | SD + CLIP | Minimal filtering, no pre-check |

### Safety Decisions

| Decision | Description |
|----------|-------------|
| `SAFE` | All checks passed, original image saved |
| `BLOCKED` | Pre-check failed, no image generated |
| `UNSAFE_BLURRED` | Post-check failed, blurred image saved |

## Project Structure

```
SafetymultifilteringSD/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration, thresholds, model IDs
├── models/
│   ├── __init__.py
│   ├── safety_agent.py      # Pre-generation LLM agent (Qwen2.5-7B)
│   ├── diffusion.py         # Stable Diffusion wrapper (Deliberate v5)
│   ├── clip_embedder.py     # CLIP ViT-L-14 baseline
│   ├── coca_embedder.py     # CoCa captioning & embedding
│   ├── nudenet_checker.py   # NudeNet v3 nudity detection
│   └── vlm_checker.py       # Qwen2-VL-7B Vision-Language Model
├── pipeline/
│   ├── __init__.py
│   └── safety_pipeline.py   # Main orchestrator (Defense-in-Depth)
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Blur, save, resize operations
│   └── metrics.py           # Benchmark timing & statistics
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Unit tests
├── outputs/
│   └── benchmark_results_*/ # Timestamped benchmark outputs
│       ├── safe/            # Safe images
│       ├── original/        # Unsafe originals (unblurred)
│       ├── blurred/         # Unsafe images (blurred)
│       ├── progress.jsonl   # Checkpoint file for resume
│       └── benchmark_report.xlsx
├── main.py                  # CLI entry point (Typer + Rich)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md
```

## Configuration

Edit `config/settings.py` to customize:

```python
# Model selection
models.safety_agent_id = "Qwen/Qwen2.5-7B-Instruct"       # Pre-generation LLM
models.diffusion_id = "stablediffusionapi/deliberate-v5"    # Image generator
models.vlm_id = "Qwen/Qwen2-VL-7B-Instruct"               # Vision-Language Model

# Quantization (4bit recommended for 8GB VRAM)
models.safety_agent_quantization = "4bit"  # 4bit, 8bit, or none
models.vlm_quantization = "4bit"

# Safety thresholds
thresholds.nudenet_threshold = 0.6   # NudeNet confidence threshold
thresholds.embedding_threshold = 0.25  # CLIP similarity threshold

# Generation parameters
generation.num_inference_steps = 70
generation.guidance_scale = 4.0
generation.width = 512
generation.height = 512

# Pipeline toggles
pipeline.enable_pre_check = True   # LLM prompt check
pipeline.enable_coca = True        # CoCa captioning
pipeline.enable_nudenet = True     # Nudity detection
pipeline.enable_vlm = True         # VLM semantic analysis
pipeline.unload_after_use = True   # Free VRAM after each model
```

## Benchmark Metrics

The pipeline tracks:
- **Latency**: Per-layer timing in milliseconds (pre_check, generation, coca, nudenet, vlm)
- **Throughput**: Images per second
- **Safety Statistics**: Count of safe, blocked, and blurred outputs
- **Layer Breakdown**: Min/max/mean timing for each pipeline stage
- **NudeNet Exposed Regions**: Which body parts triggered detection (e.g. `FEMALE_BREAST_EXPOSED`)

Results are exported to Excel with 3 sheets: Results, Summary, and Layer_Breakdown. Progress is checkpointed to `progress.jsonl` after each prompt for crash recovery.

## VRAM Usage

| Component | VRAM | Notes |
|-----------|------|-------|
| Safety Agent (Qwen2.5-7B) | ~4-5GB | 4-bit quantized |
| Stable Diffusion (Deliberate v5) | ~3.5GB | FP16 + xformers |
| CoCa ViT-L-14 | ~1.5GB | Image captioning |
| NudeNet v3 | ~300MB | Lightweight CNN |
| VLM (Qwen2-VL-7B) | ~5-6GB | 4-bit quantized |
| CLIP ViT-L-14 | ~1.5GB | Baseline mode only |

**Total Peak**: ~6-7GB (with sequential loading and auto-unload)

> **Note**: Models are loaded sequentially and unloaded after use (`unload_after_use=True` by default). This allows running all components on 8GB VRAM GPUs.

## References

- [Safe Latent Diffusion](https://arxiv.org/abs/2211.05105) - Schramowski et al., 2023
- [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [CoCa](https://arxiv.org/abs/2205.01917) - Yu et al., 2022
- [NudeNet](https://github.com/notAI-tech/NudeNet) - Nudity detection CNN
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) - Pre-generation safety agent
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) - Vision-Language Model
- [Deliberate v5](https://huggingface.co/stablediffusionapi/deliberate-v5) - Image generation

## License

MIT License - See [LICENSE](LICENSE) file.

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{karabillioglu2026nsfw,
  title={NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach},
  author={Karabillioglu, Berk},
  school={Yeditepe University},
  year={2026},
  type={Master's Thesis}
}
```

## Disclaimer

This project is for research purposes only. The authors do not endorse the generation of inappropriate content. Use responsibly and in compliance with applicable laws and regulations.
