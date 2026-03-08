# R3D-18 3D CNN Study

Systematic evaluation of R3D-18 for per-rep push-up form classification (good vs bad).

## Dataset

- **357 reps** (165 good, 192 bad) from **238 unique videos**
- Mean 1.5 reps per video; rep length: min=11, max=396, median=63 frames
- All participants performed both good and bad reps
- Reps manually annotated with human-labeled start/end frame boundaries
- 6 reps skipped due to missing video files

## Model Architecture

- **R3D-18** (3D ResNet-18) pretrained on Kinetics-400
- Input: `(B, 3, 16, 112, 112)` — 16 uniformly sampled RGB frames per rep
- Output: `(B, 2)` — binary logits (good / bad)
- Training: 5-fold stratified cross-validation, split by video to prevent data leakage
- Augmentation: horizontal flip + color jitter on training set

## Study Design

Five stages, each building on the previous winner:

1. **Experiment A**: Full-frame input (frozen backbone)
2. **Experiment B**: YOLO-crop input (frozen backbone)
3. **Unfreezing**: Unfreeze last 1 or 2 backbone blocks on the winner
4. **HP tuning**: Grid search over LR and batch size on the best unfreezing config
5. **Overfitting analysis**: Loss curves, confusion matrices, per-video errors

## Results

### Input Preprocessing (A vs B)

| Experiment | Input | Trainable Params | Mean Accuracy | Std |
|---|---|---|---|---|
| A: Full-frame (frozen) | Resize 128x171, center crop 112x112 | 1,026 | 75.2% | 4.5% |
| **B: YOLO-crop (frozen)** | YOLO bbox crop, resize 112x112 | 1,026 | **77.9%** | 4.9% |

YOLO-crop wins by ~2.7%. Cropping to the person removes irrelevant background and gives the model a cleaner signal.

### Unfreezing (on YOLO-crop)

| Config | Trainable Params | Mean Accuracy | Std |
|---|---|---|---|
| Frozen (FC only) | 1,026 | 77.9% | 4.9% |
| Unfreeze 1 block (layer4) | 24,909,826 | 86.8% | 3.0% |
| **Unfreeze 2 blocks (layer3+4)** | 31,138,306 | **89.9%** | 2.9% |

Unfreezing layer3+layer4 gives the biggest improvement (+12% over frozen). The Kinetics-400 features are useful but not perfectly suited to push-up form — fine-tuning the deeper layers helps the model learn task-specific spatiotemporal features.

### Hyperparameter Tuning (unfreeze_2 + YOLO-crop)

| LR | Batch Size | Mean Accuracy | Std |
|---|---|---|---|
| 1e-4 | 4 | 88.3% | 3.5% |
| 1e-4 | 8 | 89.9% | 2.9% |
| 1e-4 | 16 | 88.5% | 3.4% |
| **5e-4** | **4** | 92.2% | 3.1% |
| **5e-4** | **8** | **93.9%** | **2.6%** |
| 5e-4 | 16 | 92.7% | 3.0% |
| 1e-3 | 4 | 91.5% | 3.2% |
| 1e-3 | 8 | 92.4% | 2.8% |
| 1e-3 | 16 | 91.1% | 3.3% |

Best: **lr=5e-4, batch_size=8 → 93.9% +/- 2.6%**

### Final Comparison (all configs)

| Experiment | Input | Backbone | Params | LR | BS | Mean Acc | Std |
|---|---|---|---|---|---|---|---|
| A: Full-frame (frozen) | Full-frame | Frozen | 1,026 | 1e-3 | 8 | 75.2% | 4.5% |
| B: YOLO-crop (frozen) | YOLO-crop | Frozen | 1,026 | 1e-3 | 8 | 77.9% | 4.9% |
| Unfreeze 1 block (crop) | Crop | layer4 | 24.9M | 1e-4 | 8 | 86.8% | 3.0% |
| Unfreeze 2 blocks (crop) | Crop | layer3+4 | 31.1M | 1e-4 | 8 | 89.9% | 2.9% |
| **Best HP (unfreeze_2, crop)** | **Crop** | **layer3+4** | **31.1M** | **5e-4** | **8** | **93.9%** | **2.6%** |

## Overfitting Analysis

### Train vs Val Loss Curves

- **Frozen (B)**: Train-val loss gap ~0.20, moderate noise — healthy, minimal overfitting
- **Best HP (unfreeze_2)**: Train loss drops to ~0.01, val loss stays 0.3-0.8 — clear overfitting
- 31M params on ~280 training reps = very high parameter-to-sample ratio
- Val loss noisy due to small fold sizes (~70 reps per fold)
- **True generalization accuracy likely 85-90%** (93.9% CV is inflated by overfitting)

### Confusion Matrices

- Frozen models: roughly balanced errors across good and bad classes
- Best HP: errors biased toward misclassifying good reps as bad (16 good misclassified vs 6 bad)

### Per-Video Error Analysis

- 220/238 videos classified perfectly (0 errors)
- 18 videos had at least one misclassified rep
- Errors spread across videos rather than concentrated on a few

## Inference Pipelines

Two working inference modes:

### 1. Automatic (`infer_automatic`)
- Input: pre-recorded video file
- Pipeline: YOLO keypoint extraction → state machine rep detection → R3D classification
- Use case: batch processing recorded videos

### 2. Record & Classify (`live_demo.py`)
- Input: live webcam feed
- Pipeline: record video → YOLO keypoints → state machine → R3D classification
- Controls: S = start/stop recording, Q/ESC = quit
- Run from terminal: `python live_demo.py`

### State Machine

The push-up state machine tracks elbow angle transitions to detect individual reps:
- **UP → GOING_DOWN → DOWN → GOING_UP → UP** = one completed rep
- Thresholds: down=90 degrees, up=160 degrees (configurable)
- Filters degenerate reps (< 10 frames)

## Live Demo Limitations

### The Problem

The R3D model works well on pre-recorded videos processed offline (8/10 correct on test videos), but struggles with live webcam input. Testing showed:
- **Good.MOV** (pre-recorded test): 3/5 reps correctly classified as GOOD
- **Bad.MOV** (pre-recorded test): 5/5 reps correctly classified as BAD
- **Live webcam**: most reps classified as BAD regardless of actual form

### Root Cause: Domain Gap

The R3D operates on raw pixel crops (112x112 RGB frames). Any difference between training and inference conditions causes a distribution shift:

- **Camera angle/distance**: Training videos were recorded at specific angles; webcam angle differs
- **Lighting/background**: Different environments produce different pixel distributions
- **Crop quality**: YOLO bounding box stability varies between offline batch processing (clean) and live frame-by-frame (noisier)
- **Resolution/compression**: Webcam capture vs training video files have different characteristics

The model learns pixel-level features (textures, colors, edges in the crop) that are sensitive to these environmental factors. Even though the model learned some genuine form features (it partially works on test videos), those features are fragile when the visual context shifts.

### What Doesn't Work

Three live approaches were tested and discarded:

1. **Real-time state machine + R3D**: Per-frame YOLO crops were too noisy, all reps classified as BAD
2. **Sliding window**: Classifying arbitrary 2-second windows produced 55 classifications for 5 reps — too noisy, partial reps confuse the model
3. **Stable bounding box**: Using the best keypoint frame's bbox for all frames in a rep — improved crops but still all BAD

### Why Skeleton-Based Models (ST-GCN) Would Be Better

| | R3D-18 (pixels) | ST-GCN (skeleton) |
|---|---|---|
| Input | RGB crop (112x112) | Joint coordinates (12 joints x xy) |
| Sees appearance? | Yes | No — only joint positions |
| Camera angle sensitivity | High | Low (skeleton can be normalized) |
| Crop quality dependency | High | None — no crops needed |
| Domain gap risk | High | Low |
| Real-time feasibility | Needs stable crops | YOLO already extracts keypoints per frame |

A skeleton-based model only sees elbow angles, back alignment, and hip sag — exactly the features that define good vs bad form. It cannot learn spurious pixel-level shortcuts and should generalize better across camera setups.

## File Structure

```
r3d_study/
├── notebook.ipynb          # Main experiment notebook
├── model.py                # PushUpR3D model definition
├── datasets.py             # Dataset classes (full-frame, YOLO-crop, precomputed)
├── training.py             # K-fold CV training loop
├── data_loader.py          # Annotation loading + YOLO keypoint extraction
├── state_machine.py        # Push-up state machine (self-contained)
├── inference.py            # Inference pipelines (automatic + record mode)
├── live_demo.py            # Standalone webcam demo script
├── annotate_helper.py      # Frame viewer for manual annotation
├── test_r3d_study.py       # Unit tests
├── R3D_STUDY.md            # This document
└── outputs/                # Generated outputs (gitignored)
    ├── r3d_best.pt         # Best model checkpoint
    ├── r3d_study_results.csv
    ├── r3d_hp_grid.csv
    └── *.png               # Plots (comparison, heatmap, loss curves, etc.)
```

## Key Takeaways

1. **YOLO-crop > full-frame** for push-up classification — removing background noise helps
2. **Fine-tuning deeper layers is critical** — frozen backbone (77.9%) vs unfreeze 2 blocks (93.9%)
3. **Overfitting is significant** with 31M params on 357 reps — frozen baseline (77.9%) is the most trustworthy result
4. **Pixel-based models have domain gap issues** that limit live demo reliability
5. **Skeleton-based models (ST-GCN)** are better suited for real-time deployment due to invariance to visual appearance
