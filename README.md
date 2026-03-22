# NSFW Safety Filtering Pipeline for Stable Diffusion

A multi-layered safety filtering framework for text-to-image generation, implementing a **Defense-in-Depth** architecture. This project is part of a Master's thesis on **NSFW Detection and Text Filtering for Safe Diffusions: A Turkish Language Approach**.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Modular Defense-in-Depth Safety Pipeline                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────┐    ┌─────────────────┐         ┌──────────────────┐            │
│  │  Input   │───►│  Pre-Generation │──Safe?──►│ Stable Diffusion │            │
│  │  Prompt  │    │  Agent (Qwen)   │         │   (Deliberate v5) │            │
│  └──────────┘    └────────┬────────┘         └────────┬─────────┘            │
│                           │                           │                       │
│                      [BLOCKED]                        ▼                       │
│                      if unsafe              ┌──────────────────┐             │
│                                             │  Generated Image  │             │
│                                             └────────┬─────────┘             │
│                                                      │                       │
│              ┌───────────────────────────────────────┤                       │
│              │         Post-Generation Checks         │  (her biri bağımsız) │
│              │  ┌──────┐ ┌────────┐ ┌──────┐ ┌─────┐ │                       │
│              │  │ CLIP │ │  CoCa  │ │NudeN.│ │ VLM │ │                       │
│              │  │(opt.)│ │+recheck│ │      │ │     │ │                       │
│              │  └──┬───┘ └───┬────┘ └──┬───┘ └──┬──┘ │                       │
│              └─────┼─────────┼──────────┼────────┼────┘                       │
│                    └─────────┴──────────┴────────┘                            │
│                                    │                                          │
│                              Hepsi Safe?                                      │
│                             /           \                                     │
│                           Evet          Hayır                                 │
│                            │              │                                   │
│                     ┌──────▼──────┐ ┌─────▼────────┐                         │
│                     │ Safe Output │ │Blurred Output│                          │
│                     └─────────────┘ └──────────────┘                         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Modüler Yapı

Her bileşen `config/settings.py` içindeki `enable_*` flag'leri ile **bağımsız** olarak açılıp kapatılabilir. CLI `--mode` parametresi bu flag'leri toplu ayarlayan bir kısayoldur.

```
CLI --mode  →  PipelineConfig preset  →  enable_* flags  →  Pipeline
```

## Features

- **Pre-Generation Safety**: LLM-based prompt analysis using Qwen2.5-7B-Instruct (4-bit quantized)
- **Image Generation**: Stable Diffusion (Deliberate v5) with native safety checker disabled
- **Post-Generation Analysis** (her biri bağımsız):
  - **CLIP** ViT-L-14: Unsafe concept similarity check, threshold ile tüm class'lar kontrol edilir (`enable_clip=True` ile aktif)
  - **CoCa** (Contrastive Captioners): Image-to-text captioning + Agent re-check
  - **NudeNet v3**: CNN-based nudity detection with bounding boxes and exposed region reporting
  - **Qwen2-VL-7B**: Vision Language Model for semantic safety analysis
- **Three Run Modes**: `generate` (prompt→görsel), `check` (hazır görsel), `benchmark`
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
- CLIP ViT-L-14 (~1GB) - `enable_clip=True` yapılırsa indirilir

## Usage

### generate — Prompt'tan Görsel Üret

```bash
# Varsayılan (full mod): Agent + CoCa + NudeNet + VLM
python main.py generate "A beautiful sunset over the ocean"

# Seed ve boyut ile
python main.py generate "A cat playing with yarn" --seed 42 -W 768 -H 768

# Baseline mod: Agent + CLIP (daha hızlı)
python main.py generate "A mountain landscape" --mode baseline

# Sadece CLIP (pre-check yok)
python main.py generate "Abstract art" --mode clip_only

# Kaydetme
python main.py generate "A robot reading a book" --no-save
```

### check — Hazır Görseli Analiz Et

Generate olmadan, var olan bir görseli post-checker'lardan geçirir.
`pre_check` ve `generation` her zaman atlanır.

```bash
# Tek görsel — varsayılan full mod
python main.py check outputs/image.png

# Klasördeki tüm görseller — baseline (CLIP) ile
python main.py check outputs/safe/ --mode baseline

# Farklı çıktı klasörü
python main.py check outputs/test.jpg --mode full -o outputs/checked/

# Kaydetmeden sadece sonucu gör
python main.py check outputs/image.png --no-save
```

### check-prompt — Sadece Prompt Güvenliği

```bash
python main.py check-prompt "Your prompt here"
```

### benchmark — Toplu İşlem

#### Prompt Listesinden Görsel Üretimi + Post-Check

```bash
# Varsayılan örnekler
python main.py benchmark

# Excel / CSV dosyasından
python main.py benchmark -i prompts.xlsx
python main.py benchmark -i prompts.csv

# İlk 20 örnek, baseline mod
python main.py benchmark -i prompts.xlsx -n 20 --mode baseline

# Özel rapor yolu
python main.py benchmark -i prompts.xlsx -o my_report.xlsx
```

#### Hazır Görsel Setine Post-Check (`--source-dir`)

Generate olmadan, mevcut bir görsel klasörüne sadece post-checker'ları uygular.

```bash
# Klasördeki tüm görseller — full mod
python main.py benchmark --source-dir outputs/images/ --mode full

# İlk 50 görsel — CLIP ile
python main.py benchmark --source-dir outputs/images/ -n 50 --mode baseline

# Özel rapor
python main.py benchmark --source-dir outputs/raw/ -o post_check_report.xlsx
```

#### Resume (Checkpoint Desteği)

```bash
# Benchmark başlat (outputs/benchmark_results_20260222_143015/ oluşturur)
python main.py benchmark -i dataset.xlsx -n 500

# Yarıda kesilirse devam et
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 -i dataset.xlsx -n 500

# --source-dir ile resume
python main.py benchmark --resume outputs/benchmark_results_20260222_143015 --source-dir outputs/images/
```

### Benchmark Output Structure

```
outputs/benchmark_results_20260222_143015/
├── safe/
│   ├── 0001.png              # Güvenli görseller (orijinal)
│   └── 0004.png
├── original/
│   ├── 0002.png              # Güvensiz görseller (bulanıklaştırılmamış)
│   └── 0003.png
├── blurred/
│   ├── 0002.png              # Güvensiz görseller (bulanıklaştırılmış)
│   └── 0003.png
├── progress.jsonl             # Checkpoint (her öğe sonrası yazılır)
└── benchmark_report.xlsx      # Excel rapor
```

### Excel Report

3 sayfa içerir:

**Sheet 1 - Results:**
| index | prompt | pre_check_safe | pre_check_reason | image_path | decision | time_ms | coca_caption | clip_safe | clip_triggered | vlm_safe | vlm_reason | nudenet_safe | nudenet_exposed_regions |

**Sheet 2 - Summary:**
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

**Sheet 3 - Layer Breakdown:**
| layer | mean_ms | min_ms | max_ms | count |

## Pipeline Modes

| Mode | Components | Kullanım |
|------|------------|---------|
| `full` | Agent + CoCa + NudeNet + VLM | Maksimum güvenlik (varsayılan) |
| `baseline` | Agent + CLIP | Hızlı, CLIP tabanlı kontrol |
| `clip_only` | CLIP only | Minimal filtreleme, pre-check yok |

> **Not**: `--mode` flag'leri preset kısayollarıdır. `config/settings.py` içinde `enable_*` flag'lerini değiştirerek herhangi bir kombinasyonu oluşturabilirsiniz (örn. `enable_clip=True` + `enable_vlm=True`).

### Safety Decisions

| Karar | Açıklama |
|-------|----------|
| `SAFE` | Tüm kontroller geçti, orijinal görsel kaydedildi |
| `BLOCKED` | Pre-check başarısız, görsel üretilmedi |
| `UNSAFE_BLURRED` | Post-check başarısız, bulanık görsel kaydedildi |

## Project Structure

```
SafetymultifilteringSD/
├── config/
│   ├── __init__.py
│   └── settings.py          # Konfigürasyon, eşikler, model ID'leri, preset'ler
├── models/
│   ├── __init__.py
│   ├── safety_agent.py      # Pre-generation LLM agent (Qwen2.5-7B)
│   ├── diffusion.py         # Stable Diffusion wrapper (Deliberate v5)
│   ├── clip_embedder.py     # CLIP ViT-L-14 (enable_clip=True ile aktif)
│   ├── coca_embedder.py     # CoCa captioning & embedding
│   ├── nudenet_checker.py   # NudeNet v3 nudity detection
│   └── vlm_checker.py       # Qwen2-VL-7B Vision-Language Model
├── pipeline/
│   ├── __init__.py
│   └── safety_pipeline.py   # Ana orkestratör (run + check_image)
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Blur, save, resize
│   └── metrics.py           # Benchmark timing & statistics
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Unit tests
├── outputs/
│   └── benchmark_results_*/ # Zaman damgalı benchmark çıktıları
│       ├── safe/
│       ├── original/
│       ├── blurred/
│       ├── progress.jsonl
│       └── benchmark_report.xlsx
├── main.py                  # CLI entry point (Typer + Rich)
├── requirements.txt
├── LICENSE
└── README.md
```

## Configuration

`config/settings.py` dosyasını düzenleyerek özelleştirebilirsiniz:

```python
# Model seçimi
models.safety_agent_id = "Qwen/Qwen2.5-7B-Instruct"
models.diffusion_id = "stablediffusionapi/deliberate-v5"
models.vlm_id = "Qwen/Qwen2-VL-7B-Instruct"

# Kuantizasyon (8GB VRAM için 4bit önerilir)
models.safety_agent_quantization = "4bit"
models.vlm_quantization = "4bit"

# Güvenlik eşikleri
thresholds.nudenet_threshold = 0.6
thresholds.embedding_threshold = 0.25   # CLIP benzerlik eşiği

# Üretim parametreleri
generation.num_inference_steps = 70
generation.guidance_scale = 4.0
generation.width = 512
generation.height = 512

# Pipeline bileşen toggle'ları (bağımsız)
pipeline.enable_pre_check = True    # LLM prompt kontrolü
pipeline.enable_clip = False        # CLIP benzerlik kontrolü (varsayılan kapalı)
pipeline.enable_coca = True         # CoCa captioning + re-check
pipeline.enable_nudenet = True      # Çıplaklık tespiti
pipeline.enable_vlm = True          # VLM semantik analiz
pipeline.unload_after_use = True    # Her model sonrası VRAM'ı boşalt
```

### CLIP Hakkında

CLIP varsayılan olarak **kapalıdır** (`enable_clip = False`). Aktif etmek için:

```python
pipeline.enable_clip = True
```

Aktif edildiğinde:
- `UNSAFE_CONCEPTS` listesindeki her kavram için cosine similarity hesaplanır
- `thresholds.embedding_threshold` (varsayılan `0.25`) üzerinde olan **herhangi bir kavram** görseli `UNSAFE_BLURRED` olarak işaretler
- Sonuç tabloda `clip: CAUGHT (gore, violence)` veya `clip: PASSED` olarak görünür
- Excel raporunda `clip_safe` ve `clip_triggered` kolonlarına yazılır
- Kullandıktan sonra VRAM'dan boşaltılır (`unload_after_use=True`)

## Benchmark Metrics

Pipeline şunları takip eder:
- **Gecikme**: Her katman için ms cinsinden süre (pre_check, generation, clip, coca, nudenet, vlm)
- **Verimlilik**: Saniyede görsel sayısı
- **Güvenlik İstatistikleri**: Güvenli, engellenen ve bulanıklaştırılan görsel sayıları
- **Katman Dökümü**: Her aşama için min/max/ortalama süre
- **NudeNet Bölgeleri**: Tespit edilen vücut bölgeleri
- **CLIP Tetikleyenler**: Eşiği geçen kavram listesi

## VRAM Usage

| Bileşen | VRAM | Notlar |
|---------|------|--------|
| Safety Agent (Qwen2.5-7B) | ~4-5GB | 4-bit kuantize |
| Stable Diffusion (Deliberate v5) | ~3.5GB | FP16 + xformers |
| CoCa ViT-L-14 | ~1.5GB | Görsel captioning |
| NudeNet v3 | ~300MB | Hafif CNN |
| VLM (Qwen2-VL-7B) | ~5-6GB | 4-bit kuantize |
| CLIP ViT-L-14 | ~1.5GB | enable_clip=True olunca yüklenir |

**Toplam Zirve**: ~6-7GB (sıralı yükleme ve auto-unload ile)

> **Not**: Modeller sıralı yüklenir ve kullanım sonrası boşaltılır (`unload_after_use=True`). Bu sayede tüm bileşenler 8GB VRAM'lı GPU'larda çalışabilir.

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
