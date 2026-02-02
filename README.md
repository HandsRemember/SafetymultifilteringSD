# NSFW Safety Filtering Pipeline for Stable Diffusion

A multi-layered safety filtering framework for text-to-image generation, implementing a "Defense-in-Depth" architecture. This project is part of a Master's thesis on **NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Safety Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Input Prompt] ──► [Pre-Generation Agent] ──► Safe? ──► [Stable Diffusion] │
│                            │                    │                │           │
│                            │                    │                ▼           │
│                            │                    │         [Generated Image]  │
│                            │                    │                │           │
│                            ▼                    │                ▼           │
│                        [Blocked]                │      [CoCa Captioning]     │
│                            │                    │                │           │
│                            │                    │                ▼           │
│                            │                    │      [Agent Re-check]      │
│                            │                    │                │           │
│                            │                    │                ▼           │
│                            │                    │    [NudeNet] + [VLM]       │
│                            │                    │                │           │
│                            │                    │                ▼           │
│                            │                    │         Safe? ──► [Output] │
│                            │                    │           │                │
│                            │                    │           ▼                │
│                            │                    │      [Blurred Image]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Pre-Generation Safety**: LLM-based prompt analysis using Phi-3-mini (4-bit quantized)
- **Image Generation**: Stable Diffusion 1.5 with native safety checker disabled
- **Post-Generation Analysis**:
  - CoCa (Contrastive Captioners) for image-to-text captioning
  - NudeNet for nudity detection
  - Vision Language Model (Qwen2-VL) for semantic safety analysis
- **Benchmark Modes**: Compare CLIP baseline vs full CoCa pipeline
- **VRAM Optimized**: Designed for RTX 3070 (8GB VRAM)

## Requirements

- Python 3.10+
- CUDA 11.8+ compatible GPU
- ~8GB VRAM (RTX 3070 or equivalent)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SafetyFilteringSD.git
cd SafetyFilteringSD
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

The first run will download required models (~10GB total):
- Stable Diffusion 1.5 (~4GB)
- Phi-3-mini-4k-instruct (~2GB)
- CoCa ViT-L-14 (~1.5GB)
- Qwen2-VL-2B-Instruct (~2GB)
- NudeNet (~200MB)

## Usage

### Basic Generation

```bash
python main.py generate "A beautiful sunset over the ocean"
```

### With Options

```bash
# Specify seed for reproducibility
python main.py generate "A cat playing with yarn" --seed 42

# Use baseline CLIP mode (faster)
python main.py generate "A mountain landscape" --mode baseline

# Enable benchmark metrics
python main.py generate "A robot reading a book" --benchmark
```

### Batch Processing

Create a JSON file with prompts:

```json
{
  "prompts": [
    "A sunset over mountains",
    "A cat sleeping on a couch",
    "Abstract geometric art"
  ],
  "seeds": [42, 123, 456]
}
```

Run batch processing:

```bash
python main.py batch prompts.json --benchmark
```

### Prompt Safety Check (No Generation)

```bash
python main.py check-prompt "Your prompt here"
```

### Benchmark Mode

```bash
# Default test prompts
python main.py benchmark

# From Excel file (all rows)
python main.py benchmark -i prompts.xlsx

# First 20 samples only
python main.py benchmark -i prompts.xlsx -n 20

# Custom output report name
python main.py benchmark -i prompts.xlsx -o my_report.xlsx

# Save generated images
python main.py benchmark -i prompts.xlsx --save-images
```

### Excel Report Output

Benchmark automatically generates an Excel report with 3 sheets:

**Sheet 1 - Results:**
| prompt | pre_check_safe | pre_check_reason | image_path | decision | time_ms | coca_caption | vlm_safe | vlm_reason | nudenet_safe |
|--------|----------------|------------------|------------|----------|---------|--------------|----------|------------|--------------|

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

### Input Excel Format

Must have a column containing "prompt" (case-insensitive):
- `prompt`, `Prompt`, `prompt_tr`, `English_Prompt`, etc.

## Pipeline Modes

| Mode | Components | Use Case |
|------|------------|----------|
| `full` | Agent + SD + CoCa + NudeNet + VLM | Maximum safety (default) |
| `baseline` | Agent + SD + CLIP | Faster, baseline comparison |
| `clip_only` | SD + CLIP | Minimal filtering |

## Project Structure

```
SafetyFilteringSD/
├── config/
│   └── settings.py          # Configuration and thresholds
├── models/
│   ├── safety_agent.py      # Pre-generation LLM agent
│   ├── diffusion.py         # Stable Diffusion wrapper
│   ├── clip_embedder.py     # CLIP baseline
│   ├── coca_embedder.py     # CoCa captioning
│   ├── nudenet_checker.py   # Nudity detection
│   └── vlm_checker.py       # Vision-Language Model
├── pipeline/
│   └── safety_pipeline.py   # Main orchestrator
├── utils/
│   ├── image_utils.py       # Image processing
│   └── metrics.py           # Benchmark metrics
├── outputs/                 # Generated images
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.py` to customize:

```python
# Model selection
models.safety_agent_id = "microsoft/Phi-3-mini-4k-instruct"
models.diffusion_id = "runwayml/stable-diffusion-v1-5"

# Safety thresholds
thresholds.nudenet_threshold = 0.6
thresholds.embedding_threshold = 0.25

# Generation parameters
generation.num_inference_steps = 50
generation.guidance_scale = 7.5
```

## Benchmark Metrics

The pipeline tracks:
- **Latency**: Per-layer timing in milliseconds
- **Throughput**: Images per second
- **Safety Recall**: Percentage of unsafe content caught
- **False Positive Rate**: Safe content incorrectly flagged

## VRAM Usage

| Component | VRAM | Notes |
|-----------|------|-------|
| Safety Agent (Phi-3) | ~2.5GB | 4-bit quantized |
| Stable Diffusion 1.5 | ~3.5GB | FP16 + xformers |
| CoCa ViT-L-14 | ~1.5GB | Shared with CLIP |
| NudeNet | ~200MB | Lightweight CNN |
| VLM (Qwen2-VL-2B) | ~1.5GB | 4-bit quantized |

**Total Peak**: ~6-7GB (sequential loading)

## References

- [Safe Latent Diffusion](https://arxiv.org/abs/2211.05105) - Schramowski et al., 2023
- [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [CoCa](https://arxiv.org/abs/2205.01917) - Yu et al., 2022
- [NudeNet](https://github.com/notAI-tech/NudeNet)

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
