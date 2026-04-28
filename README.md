# Smart Vision System for Reflective and Transparent Object Understanding

A production-quality, modular computer vision pipeline for detecting, segmenting, and tracking **difficult objects** — glass, plastic, reflective surfaces, and semi-transparent materials — using a fusion of deep learning and classical CV techniques.

---

## Problem Motivation

Standard object detectors struggle with:
- **Transparent objects** (glass cups, windows, plastic bags) — lack of distinct color/texture
- **Reflective surfaces** (mirrors, metal) — environment-dependent appearance
- **Semi-transparent objects** — visual bleed-through confuses region proposals

This system addresses these challenges by fusing multiple signals: deep detection (YOLOv8), precise segmentation (SAM), and classical CV cues (edges, highlights, transparency estimation).

---

## Architecture

```
Input (Image / Video / Webcam)
         │
         ▼
┌─────────────────┐
│  YOLOv8 Detector│  ← Phase 3: Bounding box + class + confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SAM Segmentor  │  ← Phase 4: Per-object pixel masks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CV Enhancer    │  ← Phase 5: Edges, highlights, transparency map
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fusion Engine  │  ← Phase 6: Confidence-weighted combination
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SORT Tracker   │  ← Phase 7: Kalman filter + Hungarian matching
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Depth (opt.)   │  ← Phase 8: DPT monocular depth / simulated
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualizer     │  ← Phase 10: Boxes, masks, IDs, stats overlay
└────────┬────────┘
         │
         ▼
   Output (Image / Video)
```

---

## Project Structure

```
reflective_vision_system/
├── configs/
│   ├── config.yaml          # Master config (all hyperparams)
│   └── dataset.yaml         # YOLO dataset spec
├── data/
│   ├── raw/                 # Original downloaded images
│   ├── processed/           # YOLO-format split dataset
│   └── synthetic/           # Programmatically generated data
├── models/                  # Model weights (SAM, YOLO)
├── outputs/
│   ├── images/              # Output frames
│   ├── videos/              # Output video clips
│   └── logs/                # Run logs + eval metrics
├── src/
│   ├── dataset.py           # Dataset loader + augmentations
│   ├── synthetic_gen.py     # Synthetic data generator
│   ├── train.py             # YOLOv8 training
│   ├── detector.py          # YOLOv8 inference
│   ├── evaluate.py          # Detection evaluation
│   ├── segmentation.py      # SAM segmentation
│   ├── cv_enhancement.py    # Classical CV features
│   ├── fusion_engine.py     # Multi-module fusion
│   ├── tracking.py          # SORT object tracker
│   ├── depth.py             # Monocular depth estimation
│   └── visualizer.py        # Output rendering
├── utils/
│   ├── config.py            # YAML config loader
│   └── logger.py            # Rich + file logging
├── main.py                  # End-to-end pipeline entry point
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Setup

### Requirements
- Python 3.10+
- CUDA 11.8+ (recommended) or CPU

### Install

```bash
git clone https://github.com/your-username/reflective_vision_system.git
cd reflective_vision_system
bash setup.sh
source venv/bin/activate
```

`setup.sh` will:
1. Create a virtual environment
2. Install all dependencies from `requirements.txt`
3. Download the SAM ViT-H checkpoint (~2.4 GB)

---

## Usage

### 1. Generate Synthetic Training Data
```bash
python src/synthetic_gen.py --num_images 1000 --output data/processed
```

### 2. Train the Detector
```bash
python src/train.py --config configs/config.yaml
```

### 3. Evaluate
```bash
python src/evaluate.py --weights models/reflective_detector/weights/best.pt
```

### 4. Run Full Pipeline

```bash
# On a single image
python main.py --source data/raw/sample.jpg --weights models/reflective_detector/weights/best.pt

# On a video
python main.py --source data/raw/sample.mp4 --depth

# On webcam
python main.py --source 0

# Quick demo (generates synthetic frames, no weights needed)
python main.py --demo
```

### 5. Run Individual Modules

```bash
# CV Enhancement only
python src/cv_enhancement.py --image data/raw/sample.jpg

# Segmentation only
python src/segmentation.py --image data/raw/sample.jpg --weights models/sam_vit_h_4b8939.pth

# Detection only
python src/detector.py --source data/raw/sample.jpg --weights models/reflective_detector/weights/best.pt
```

---

## Target Classes

| ID | Class | Challenge |
|----|-------|-----------|
| 0 | glass_bottle | Transparent, reflective highlights |
| 1 | plastic_container | Semi-transparent, variable color |
| 2 | reflective_surface | Mirror-like, environment-dependent |
| 3 | semi_transparent_object | Background visible through object |
| 4 | glass_cup | Clear, specular glare |
| 5 | plastic_bag | Nearly invisible edges |
| 6 | mirror | Perfect reflection, no texture |
| 7 | window | Large, transparent with frame |

---

## Recommended Datasets

| Dataset | Objects | Link |
|---------|---------|------|
| Trans10K | Transparent objects | [GitHub](https://github.com/xieenze/Trans10K) |
| GSD | Glass surfaces | Research |
| COCO (subset) | Bottles, cups, wine glasses | [cocodataset.org](https://cocodataset.org) |
| OpenImages | Plastic, mirrors | [Google](https://storage.googleapis.com/openimages) |
| Synthetic (built-in) | All 8 classes | `src/synthetic_gen.py` |

---

## Fusion Strategy

Each detection gets a fused confidence score:

```
fused_conf = 0.5 × det_conf + 0.3 × seg_conf + 0.2 × cv_conf
```

Where:
- `det_conf` — YOLOv8 objectness score
- `seg_conf` — SAM mask coverage inside detection box
- `cv_conf` — Edge density + highlight presence + transparency map

Detections below `min_confidence: 0.3` are discarded.

---

## Future Improvements

- [ ] Replace SORT with DeepSORT (appearance embeddings for re-ID)
- [ ] Add polarization camera support for better reflection analysis
- [ ] Fine-tune SAM on transparent object datasets
- [ ] Export to ONNX / TensorRT for real-time edge deployment
- [ ] Add instance matting for precise alpha channel extraction
- [ ] Stereo depth instead of monocular estimation
- [ ] Active learning loop for hard example mining
- [ ] REST API wrapper (FastAPI) for serving predictions

---

## License

MIT License. See `LICENSE` for details.
