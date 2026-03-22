# NSFW Safety Filtering Pipeline for Stable Diffusion

A multi-layered safety filtering framework for text-to-image generation, implementing a **Defense-in-Depth** architecture. This project is part of a Master's thesis on **NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Modular Defense-in-Depth Safety Pipeline                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────────┐         ┌──────────────────┐           │
│  │  Input   │───►│ Pre-Generation   │──Safe?──►│ Stable Diffusion │          │
│  │  Prompt  │    │ Agent (Qwen2.5)  │         │  (Deliberate v5)  │          │
│  └──────────┘    └────────┬─────────┘         └────────┬─────────┘           │
│                           │                            │                     │
│                      [BLOCKED]                         ▼                     │
│                      if unsafe               ┌──────────────────┐            │
│                                              │  Generated Image  │           │
│                                              └────────┬─────────┘            │
│                                                       │                      │
│             ┌─────────────────────────────────────────┤                      │
│             │          Post-Generation Checks         │   (each independent) │
│             │  ┌──────┐  ┌────────┐  ┌───────┐ ┌────┐ │                      │
│             │  │ CLIP │  │  CoCa  │  │NudeNet│ │ VLM│ │                      │
│             │  │      │  │+recheck│  │       │ │    │ │                      │
│             │  └──┬───┘  └───┬────┘  └───┬───┘ └─┬──┘ │                      │
│             └─────┼──────────┼───────────┼───────┼────┘                      │
│                   └──────────┴───────────┴───────┘                           │
│                                    │                                         │
│                               All Safe?                                      │
│                             /           \                                    │
│                           Yes            No                                  │
│                            │              │                                  │
│                    ┌───────▼──────┐ ┌─────▼────────┐                         │
│                    │ Safe Output  │ │Blurred Output│                         │
│                    └──────────────┘ └──────────────┘                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Modular Design

Every component is **independently** toggled via `enable_*` flags in `config/settings.py`. The CLI `--mode` argument is a shortcut that sets these flags as a group (preset). Omitting `--mode` uses `settings.py` flags directly.

```
CLI --mode  →  PipelineConfig preset  →  enable_* flags  →  Pipeline
(omitted)   →  settings.pipeline (read enable_* directly)
```

---

## Features

- **Pre-Generation Safety**: LLM-based prompt analysis using Qwen2.5-7B-Instruct (4-bit quantized)
- **Image Generation**: Stable Diffusion (Deliberate v5) with native safety checker disabled
- **Post-Generation Analysis** (each component independently controllable):
  - **CLIP** ViT-L-14: Concept similarity check against 22 unsafe concepts derived from Rando et al. (2022) reverse-engineered SD v1.4 safety filter; per-concept thresholds; active when `enable_clip=True`
  - **CoCa** ViT-L-14: Image-to-text captioning followed by agent safety re-check
  - **NudeNet v3**: CNN-based nudity detection with bounding boxes and exposed region classification
  - **Qwen2-VL-7B**: Vision-Language Model for semantic safety reasoning
- **Three Execution Modes**: `generate` (prompt → image), `check` (existing image), `benchmark` (batch)
- **VRAM Optimized**: Sequential model loading with auto-unload for 8GB VRAM GPUs (RTX 3070)
- **Benchmark System**: Excel/CSV input, Excel reports with per-layer timing, checkpoint/resume support

---

## Requirements

- Python 3.10+
- CUDA 11.8+ compatible GPU
- ~8GB VRAM (RTX 3070 or equivalent)
- ~15GB disk space for model downloads

---

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

# Linux / Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. First Run (Model Download)

Models are downloaded automatically on first use (~15GB total):

| Model | Size | Purpose |
|-------|------|---------|
| Deliberate v5 | ~4GB | Image generation |
| Qwen2.5-7B-Instruct | ~5GB | Pre-generation safety agent (4-bit) |
| CoCa ViT-L-14 | ~1.5GB | Image captioning |
| Qwen2-VL-7B-Instruct | ~5GB | Vision-Language Model (4-bit) |
| NudeNet v3 | ~200MB | Nudity detection |
| CLIP ViT-L-14 | ~1GB | Loaded only when `enable_clip=True` |

---

## Usage

### generate — Generate Image from Prompt

```bash
# Default: uses settings.py enable_* flags directly
python main.py generate "A beautiful sunset over the ocean"

# With seed and dimensions
python main.py generate "A cat playing with yarn" --seed 42 -W 768 -H 768

# Explicit preset: Agent + CoCa + NudeNet + VLM
python main.py generate "A mountain landscape" --mode full

# Preset: Agent + CLIP only (faster)
python main.py generate "A mountain landscape" --mode baseline

# Preset: CLIP only, no pre-check
python main.py generate "Abstract art" --mode clip_only

# Skip saving output
python main.py generate "A robot reading a book" --no-save
```

### check — Analyze Existing Image(s)

Runs post-checkers on one or more images without generating anything.
`pre_check` and `generation` are always skipped.

```bash
# Single image — uses settings.py flags
python main.py check outputs/image.png

# Directory (all PNG/JPG) — baseline preset
python main.py check outputs/safe/ --mode baseline

# Save results to a custom directory
python main.py check outputs/test.jpg --mode full -o outputs/rechecked/

# Show results without saving
python main.py check outputs/image.png --no-save
```

### check-prompt — Prompt Safety Only

```bash
python main.py check-prompt "Your prompt here"
```

### benchmark — Batch Processing

#### Generate from Prompt List

```bash
# Built-in sample prompts
python main.py benchmark

# From Excel / CSV
python main.py benchmark -i prompts.xlsx
python main.py benchmark -i prompts.csv

# First 20 samples, baseline mode
python main.py benchmark -i prompts.xlsx -n 20 --mode baseline

# Custom report path
python main.py benchmark -i prompts.xlsx -o my_report.xlsx
```

#### Post-Check an Existing Image Set (`--source-dir`)

Skips generation entirely; runs only post-checkers on images in a directory.

```bash
# All images in directory — full mode
python main.py benchmark --source-dir outputs/images/ --mode full

# First 50 images — CLIP baseline
python main.py benchmark --source-dir outputs/images/ -n 50 --mode baseline

# Custom report
python main.py benchmark --source-dir outputs/raw/ -o post_check_report.xlsx
```

#### Resume (Checkpoint Support)

Each result is written to `progress.jsonl` immediately. If interrupted, resume with:

```bash
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 -i dataset.xlsx -n 500

# With --source-dir
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 --source-dir outputs/images/
```

---

## Pipeline Modes (`--mode`)

| Mode | Components Active | Use Case |
|------|-------------------|----------|
| *(omitted)* | Reads `settings.py` flags | Custom combination |
| `full` | Agent + CoCa + NudeNet + VLM | Maximum safety (recommended) |
| `baseline` | Agent + CLIP | Fast, embedding-based comparison |
| `clip_only` | CLIP only | Minimal filtering, no pre-check |

> **Tip**: To combine CLIP with other checkers (e.g. CLIP + VLM), set the relevant `enable_*` flags in `settings.py` and run without `--mode`.

### Safety Decisions

| Decision | Description |
|----------|-------------|
| `SAFE` | All checks passed — original image saved |
| `BLOCKED` | Pre-check failed — no image generated |
| `UNSAFE_BLURRED` | Post-check failed — blurred image saved |

---

## Output Structure

```
outputs/benchmark_results_20260222_143015/
├── safe/
│   ├── 0001.png              # Images that passed all checks
│   └── 0004.png
├── original/
│   ├── 0002.png              # Unsafe images (unblurred original)
│   └── 0003.png
├── blurred/
│   ├── 0002.png              # Unsafe images (blurred)
│   └── 0003.png
├── progress.jsonl             # Checkpoint file (one JSON line per item)
└── benchmark_report.xlsx      # Final Excel report
```

---

## Excel Report

**Sheet 1 — Results:**
| index | prompt | pre_check_safe | pre_check_reason | image_path | decision | time_ms | coca_caption | clip_safe | clip_triggered | vlm_safe | vlm_reason | nudenet_safe | nudenet_exposed_regions |

**Sheet 2 — Summary:**
| Metric | Value |
|--------|-------|
| Total Images | 100 |
| Mean Latency (ms) | 2345 |
| Final Safe | 85 |
| Final Blocked | 10 |
| Final Blurred | 5 |
| Pre-check Unsafe | 10 |
| CLIP Unsafe | 3 |
| NudeNet Unsafe | 3 |
| VLM Unsafe | 2 |

**Sheet 3 — Layer Breakdown:**
| layer | mean_ms | min_ms | max_ms | count |

---

## Configuration

Edit `config/settings.py` to customize behaviour:

```python
# Model selection
models.safety_agent_id = "Qwen/Qwen2.5-7B-Instruct"
models.diffusion_id    = "stablediffusionapi/deliberate-v5"
models.vlm_id          = "Qwen/Qwen2-VL-7B-Instruct"

# Quantization (4-bit recommended for 8GB VRAM)
models.safety_agent_quantization = "4bit"   # "4bit" | "8bit" | "none"
models.vlm_quantization          = "4bit"

# Safety thresholds
thresholds.nudenet_threshold  = 0.6    # NudeNet detection confidence
thresholds.embedding_threshold = 0.28  # CLIP global fallback threshold

# Independent pipeline stage toggles
pipeline.enable_pre_check = True    # LLM prompt check
pipeline.enable_clip      = False   # CLIP similarity check (off by default)
pipeline.enable_coca      = True    # CoCa captioning + re-check
pipeline.enable_nudenet   = True    # Nudity detection
pipeline.enable_vlm       = True    # VLM semantic analysis
pipeline.unload_after_use = True    # Free VRAM after each model

# Generation parameters
generation.num_inference_steps = 50
generation.guidance_scale      = 4.0
generation.width               = 512
generation.height              = 512
```

### CLIP Configuration

CLIP is **disabled by default** (`enable_clip = False`). To activate:

```python
pipeline.enable_clip = True
```

When active:
- Computes cosine similarity against 22 unsafe concepts (sexual content, violence/gore, child protection)
- Each concept has its own threshold in `thresholds.clip_per_class_thresholds` (based on Rando et al. 2022)
- Any concept exceeding its threshold → image marked `UNSAFE_BLURRED`
- Output shows: `CLIP Triggered: nude, naked` and `CLIP Max Similarity: 0.241 (nude)`
- Excel report includes `clip_safe` and `clip_triggered` columns
- Model is unloaded from VRAM after use (`unload_after_use=True`)

---

## VRAM Usage

| Component | VRAM | Notes |
|-----------|------|-------|
| Safety Agent (Qwen2.5-7B) | ~4–5 GB | 4-bit quantized |
| Stable Diffusion (Deliberate v5) | ~3.5 GB | FP16 + xformers |
| CoCa ViT-L-14 | ~1.5 GB | Image captioning |
| NudeNet v3 | ~300 MB | Lightweight CNN |
| VLM (Qwen2-VL-7B) | ~5–6 GB | 4-bit quantized |
| CLIP ViT-L-14 | ~1.5 GB | Loaded only when `enable_clip=True` |

**Peak usage**: ~6–7 GB with sequential loading and `unload_after_use=True`.

> Models are loaded one at a time and unloaded immediately after use, allowing all components to run on an 8 GB VRAM GPU.

---

## Project Structure

```
SafetymultifilteringSD/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration, thresholds, model IDs, presets
├── models/
│   ├── __init__.py
│   ├── safety_agent.py       # Pre-generation LLM agent (Qwen2.5-7B)
│   ├── diffusion.py          # Stable Diffusion wrapper (Deliberate v5)
│   ├── clip_embedder.py      # CLIP ViT-L-14 (active when enable_clip=True)
│   ├── coca_embedder.py      # CoCa captioning & embedding
│   ├── nudenet_checker.py    # NudeNet v3 nudity detection
│   └── vlm_checker.py        # Qwen2-VL-7B Vision-Language Model
├── pipeline/
│   ├── __init__.py
│   └── safety_pipeline.py    # Main orchestrator — run() and check_image()
├── utils/
│   ├── __init__.py
│   ├── image_utils.py        # Blur, save, resize operations
│   └── metrics.py            # Benchmark timing & statistics
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit tests
├── outputs/
│   └── benchmark_results_*/  # Timestamped benchmark output directories
│       ├── safe/
│       ├── original/
│       ├── blurred/
│       ├── progress.jsonl
│       └── benchmark_report.xlsx
├── main.py                   # CLI entry point (Typer + Rich)
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## References

- [Safe Latent Diffusion](https://arxiv.org/abs/2211.05105) — Schramowski et al., 2023
- [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610) — Rando et al., 2022
- [CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [CoCa](https://arxiv.org/abs/2205.01917) — Yu et al., 2022
- [NudeNet](https://github.com/notAI-tech/NudeNet) — Nudity detection CNN
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — Pre-generation safety agent
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) — Vision-Language Model
- [Deliberate v5](https://huggingface.co/stablediffusionapi/deliberate-v5) — Image generation model

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@mastersthesis{karabillioglu2026nsfw,
  title   = {NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach},
  author  = {Karabillioglu, Berk},
  school  = {Yeditepe University},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## Disclaimer

This project is for academic research purposes only. The authors do not endorse the generation of inappropriate content. Use responsibly and in compliance with applicable laws and regulations.
