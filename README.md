# Push-Up Tracker

Push-up form analysis and repetition counting using multi-model pose estimation.

**CS604 Group Project** | Python 3.12 | Managed with [uv](https://docs.astral.sh/uv/)

## Quick Start

### 1. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd pushup-tracker
uv sync --dev
```

This installs all runtime + dev dependencies (PyTorch, YOLO, MediaPipe, Jupyter, pytest, etc.) into a `.venv` automatically.

### 3. Get the dataset

The Kaggle push-up dataset (100 videos) is **not included in git** due to size. Download it:

```bash
# Option A: Kaggle CLI (requires ~/.kaggle/kaggle.json)
uv run kaggle datasets download -d <dataset-id> -p data/raw/kaggle_pushups --unzip

# Option B: Manual download
# Download from Kaggle, extract into:
#   data/raw/kaggle_pushups/Correct sequence/   (50 .mp4 files)
#   data/raw/kaggle_pushups/Wrong sequence/      (50 .mp4 files)
#   data/raw/kaggle_pushups/labels/              (correct.npy, incorrect.npy)
```

### 4. Verify everything works

```bash
uv run pytest tests/ -v
```

All 43 tests should pass. No GPU required — everything runs on CPU.

## Project Structure

```
pushup-tracker/
├── src/
│   ├── pose_estimation/       # Model wrappers + unified 12-joint format
│   │   ├── base.py            # PoseEstimatorBase ABC + PoseResult
│   │   ├── keypoint_schema.py # Unified joints, mapping tables
│   │   ├── yolo_estimator.py  # YOLO11s-pose wrapper
│   │   ├── mediapipe_estimator.py  # MediaPipe PoseLandmarker (Tasks API)
│   │   ├── movenet_estimator.py    # MoveNet (not active — see below)
│   │   └── visualization.py   # Skeleton drawing utilities
│   ├── features/              # Angle computation, normalization, temporal
│   ├── classification/        # Per-rep form classification (ST-GCN, 3D CNN, baseline)
│   ├── counting/              # State machine rep counter
│   ├── quality/               # RepScore scoring + text feedback
│   ├── benchmark/
│   │   ├── runner.py          # BenchmarkRunner (latency + full video)
│   │   └── extract_keypoints.py  # CLI: batch keypoint extraction
│   └── demo/                  # Live webcam demo
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset inventory + metadata
│   ├── 02_model_comparison.ipynb    # Visual comparison + stress test
│   ├── 03_benchmark_analysis.ipynb  # Quantitative benchmark + stress test
│   ├── 04_feature_engineering.ipynb    # Angle computation + normalization
│   ├── 05_rep_counting_baseline.ipynb # State machine evaluation
│   ├── 06_form_classification.ipynb   # Per-rep ST-GCN vs 3D CNN comparison
│   ├── 07–08                          # Quality assessment, final evaluation
├── configs/
│   └── quality_thresholds.yaml
├── tests/                     # 43 unit tests
├── data/                      # gitignored — see "Get the dataset" above
│   ├── raw/kaggle_pushups/    # 100 source videos
│   ├── raw/stress_test/       # Challenge videos (see below)
│   └── processed/             # Extracted keypoints + benchmark CSVs
├── models/                    # gitignored — auto-downloaded on first run
├── outputs/results/           # Summary CSVs from notebook runs
└── pyproject.toml
```

## What's Implemented (Phases 1–3)

### Pose Estimation (2 models active)

| Model | Input | Speed | Notes |
|-------|-------|-------|-------|
| **YOLO11s-pose** | Any resolution | ~25ms/frame | Auto-downloads `yolo11s-pose.pt` on first use |
| **MediaPipe PoseLandmarker** | Any resolution | ~16ms/frame | Auto-downloads `.task` bundle from Google Cloud |

Both output a **unified 12-joint format** (shoulders, elbows, wrists, hips, knees, ankles) as `(12, 3)` arrays of `[x, y, confidence]`.

### MoveNet (not active)

MoveNet Lightning/Thunder wrappers exist in `movenet_estimator.py` but are **non-functional** because `tfhub.dev` is dead and TFLite models are hard to source. A teammate can fix this by:
- Finding working download URLs (Kaggle models API, or a GitHub mirror)
- The `ai-edge-litert` package works on macOS ARM as a TFLite interpreter
- The wrapper code and mapping tables are already written

### Feature Engineering, Counting & Quality (Proof-of-Concept)

**Features** (`src/features/`)
- Joint angle computation (elbow, shoulder, hip, knee)
- Coordinate normalization (torso-length)
- Temporal features (velocity, acceleration, Savitzky-Golay smoothing)

**Counting** (`src/counting/`)
- Rule-based state machine for rep counting (4-phase elbow angle transitions)

### Form Classification (Phase 3)

Classifies each individual push-up repetition as correct or incorrect (not whole videos). The state machine segments videos into reps, and each rep is classified independently. Labels are inherited from the parent video (all reps in a "correct" video are labeled correct).

- **Logistic Regression baseline** on per-rep angle statistics (16 features)
- **3D CNN (R3D-18)**: Pretrained on Kinetics-400, fine-tuned FC layer on per-rep video clips
- **ST-GCN**: Spatial-Temporal Graph Convolutional Network on per-rep skeleton sequences
- 5-fold stratified CV — splits by video to prevent data leakage, trains/evaluates on rep-level samples

**Quality** (`src/quality/`)
- RepScore quality scorer (back alignment, depth, extension → composite 0–100) — scoring cutoffs are defaults, need validation
- Text feedback generation from RepScore

## Live Demo

Real-time push-up tracking with skeleton overlay, rep counting, phase detection, and angle readouts.

### Webcam

```bash
# YOLO (default)
uv run python -m src.demo.live

# MediaPipe
uv run python -m src.demo.live --model mediapipe
```

### Video file

```bash
# Play a dataset video with tracking overlay
uv run python -m src.demo.live --video "data/raw/kaggle_pushups/Correct sequence/Copy of push up 80.mp4"

# Use MediaPipe instead
uv run python -m src.demo.live --video path/to/video.mp4 --model mediapipe

# Save the annotated output
uv run python -m src.demo.live --video path/to/video.mp4 --save output.mp4
```

### What's shown on screen

- Skeleton overlay (color-coded left/right sides)
- **Rep count** — increments on each completed push-up
- **Phase** — UP, GOING_DOWN, DOWN, GOING_UP (color-coded)
- **Elbow angle** + **back alignment angle** (degrees)
- **Per-rep form quality** — "Rep 3: CORRECT (94%)" after each completed rep (when `--form-model` is provided)
- Inference time (ms)
- Phase indicator bar at the bottom
- Progress bar for video files

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `space` | Pause / resume |
| `r` | Reset rep count (and restart video) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo` | Pose model (`yolo`, `mediapipe`) |
| `--video` | — | Path to video file (omit for webcam) |
| `--camera` | `0` | Camera device ID for webcam mode |
| `--save` | — | Save annotated video to this path |
| `--down-threshold` | `90` | Elbow angle (degrees) for "down" position |
| `--up-threshold` | `160` | Elbow angle (degrees) for "up" position |
| `--form-model` | — | Path to trained ST-GCN model (.pt) for real-time form feedback |

The model flag is **extensible** — when new models are added to the registry in `src/demo/live.py`, they become available automatically via `--model`.

### Current limitations

The demo currently uses a **rudimentary state machine** that only tracks **elbow angle** to count reps:

```
UP (>160°) → GOING_DOWN → DOWN (<90°) → GOING_UP → UP = 1 rep counted
```

The back alignment angle is displayed on screen but **not used** for counting or quality feedback. This is a proof-of-concept — future phases should extend it to:

- **Tune thresholds**: The 90°/160° elbow angle cutoffs are initial guesses, not validated on real data (see notebook 05)
- **Multi-angle counting**: Incorporate back alignment, hip angle, and knee angle into the state machine for more robust phase detection
- **Live quality feedback**: Integrate the `QualityScorer` to show real-time form feedback (back sag warnings, depth cues, etc.)
- **Form classification**: Use `--form-model models/stgcn_best.pt` to enable real-time form feedback (requires running notebook 06 first to train the model)

---

## Running the Pipeline

### Extract keypoints from all videos

```bash
# Both models, all 100 videos (~12 min total)
uv run python -m src.benchmark.extract_keypoints --models yolo mediapipe

# Quick test: 2 videos, YOLO only
uv run python -m src.benchmark.extract_keypoints --models yolo --max-videos 2
```

This creates:
- `data/processed/keypoints/{model}/*.npy` — per-video keypoint arrays, shape `(T, 12, 3)`
- `data/processed/keypoints/manifest.json` — metadata for all videos
- `data/processed/benchmark/full_benchmark.csv` — per-frame stats

The script is **resumable** — re-running skips already-processed videos.

### Run latency benchmark

```bash
uv run python -c "
from src.pose_estimation import YoloEstimator, MediaPipeEstimator
from src.benchmark.runner import BenchmarkRunner
models = [YoloEstimator(), MediaPipeEstimator()]
runner = BenchmarkRunner(models)
runner.run_latency_benchmark(n_frames=100)
"
```

Creates `data/processed/benchmark/latency_benchmark.csv`.

### Run notebooks

#### 1. Register the kernel (one-time setup)

The notebooks need a Jupyter kernel that points to the project's virtual environment so all dependencies (PyTorch, YOLO, MediaPipe, etc.) are available.

```bash
# Install the kernel from the project's .venv
uv run python -m ipykernel install --user --name pushup-tracker --display-name "Push-Up Tracker (Python 3.12)"
```

#### 2. Launch Jupyter

```bash
uv run jupyter notebook notebooks/
```

#### 3. Select the kernel

When you open a notebook, go to **Kernel → Change kernel → Push-Up Tracker (Python 3.12)**. If the notebook was already using a different kernel, switch it — otherwise imports like `from src.pose_estimation import ...` will fail.

Run in order: 01 (no model needed) -> 02 (loads models) -> 03 (needs CSVs from extraction).

## Stress Test Videos

To test model robustness beyond the clean Kaggle data, add short clips to:

```
data/raw/stress_test/
├── side_angle/        # Profile view
├── front_angle/       # Head-on view
├── diagonal/          # ~45 degree angle
├── low_resolution/    # 240p-360p
├── poor_lighting/     # Dark / backlit
├── noisy_compressed/  # Heavy compression artifacts
└── multiple_people/   # 2+ people in frame
```

Notebooks 02 and 03 auto-detect these and include them in the analysis. Empty folders are skipped.

To download from YouTube:
```bash
uv tool install yt-dlp
yt-dlp --merge-output-format mp4 -o "data/raw/stress_test/FOLDER/%(title)s.%(ext)s" "YOUTUBE_URL"

# For a specific time range:
yt-dlp --merge-output-format mp4 --download-sections "*0:12-0:20" -o "..." "URL"

# For lowest quality (stress testing resolution):
yt-dlp -f worst --merge-output-format mp4 -o "..." "URL"
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## For Teammates: What to Work On Next

### Phase 3 — Feature Validation + Form Classification (notebooks 04–06)

- **04 — Feature Engineering**: Visualize actual angle curves from extracted keypoints, sanity-check features on real data, identify any normalization issues
- **05 — Rep Counting Baseline**: Run the state machine on real videos, tune the angle thresholds (currently 90°/160°), measure counting accuracy vs ground-truth labels
- **06 — Form Classification**: Train and compare Logistic Regression baseline vs R3D-18 (3D CNN) vs ST-GCN on per-rep correct/incorrect form labels

### Phase 4 — Quality Assessment + Final Evaluation (notebooks 07–08)

- **07 — Quality Assessment**: Run the quality scorer on real reps (correct vs incorrect), validate/tune scoring cutoffs, refine feedback messages
- **08 — Final Evaluation**: End-to-end pipeline (best model → features → counting → quality → form) on held-out videos, summary figures + report

### Other

- **MoveNet/RTMPose**: Add as a third pose model — wrappers + mapping tables are scaffolded in `movenet_estimator.py`
- **More stress test videos**: Add clips for `poor_lighting/` and `noisy_compressed/` categories (currently empty)
