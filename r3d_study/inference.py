"""Inference pipelines for push-up rep counting + form classification.

Three approaches:
  1. infer_annotated() — human-annotated rep boundaries (most accurate)
  2. infer_automatic() — state machine auto-detects reps from recorded video
  3. live_demo() — real-time webcam feed with rep counting + form classification

All use the best saved R3D model and YOLO keypoints.
"""

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import PushUpR3D
from datasets import (
    _uniform_sample_indices,
    _read_frames,
    _preprocess_full_frame,
    _preprocess_cropped_frame,
    _to_tensor,
    KINETICS_MEAN,
    KINETICS_STD,
)
from state_machine import PushUpStateMachine, compute_elbow_angle


def _get_extract_keypoints():
    """Lazy import of data_loader to avoid pulling in pandas for live demo."""
    from data_loader import extract_keypoints
    return extract_keypoints

logger = logging.getLogger(__name__)


# ============================================================
# Model loading
# ============================================================


def load_model(
    checkpoint_path: str | Path = "outputs/r3d_best.pt",
    device: str = "cpu",
) -> tuple[PushUpR3D, dict]:
    """Load the best R3D model from a checkpoint.

    Returns:
        (model, config) where config has keys: input_type, unfreeze,
        n_frames, accuracy, lr, batch_size.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PushUpR3D(freeze_backbone=True)
    unfreeze = ckpt.get("unfreeze", "frozen")
    if unfreeze == "unfreeze_1":
        model.unfreeze_last_n_blocks(1)
    elif unfreeze == "unfreeze_2":
        model.unfreeze_last_n_blocks(2)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    config = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, config


# ============================================================
# Core classification
# ============================================================


def _classify_rep(
    model: PushUpR3D,
    config: dict,
    video_path: str,
    start_frame: int,
    end_frame: int,
    keypoints: np.ndarray | None,
    device: str,
) -> tuple[str, float]:
    """Classify a single rep clip.

    Returns:
        (label_str, confidence) — "GOOD"/"BAD" and softmax probability.
    """
    n_frames = config.get("n_frames", 16)
    input_type = config.get("input_type", "full")

    indices = _uniform_sample_indices(start_frame, end_frame, n_frames)
    raw_frames = _read_frames(video_path, indices)

    if input_type == "crop" and keypoints is not None:
        processed = []
        for frame, fi in zip(raw_frames, indices):
            kp_idx = max(0, min(fi - start_frame, len(keypoints) - 1))
            processed.append(_preprocess_cropped_frame(frame, keypoints[kp_idx]))
    else:
        processed = [_preprocess_full_frame(f) for f in raw_frames]

    tensor = torch.from_numpy(_to_tensor(processed)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        confidence = probs[0, pred].item()

    label = "GOOD" if pred == 0 else "BAD"
    return label, confidence


# ============================================================
# Approach 1: Annotated rep boundaries
# ============================================================


def infer_annotated(
    video_path: str | Path,
    reps: list[dict],
    model: PushUpR3D,
    config: dict,
    keypoint_dir: str | Path = "keypoints",
    device: str = "cpu",
) -> list[dict]:
    """Classify pre-annotated reps from a video.

    Args:
        video_path: Path to the video file.
        reps: List of dicts, each with "start_frame", "end_frame",
              and optionally "expected" ("good"/"bad").
        model: Loaded R3D model.
        config: Model config from checkpoint.
        keypoint_dir: Directory for YOLO keypoint .npy files.
        device: Torch device string.

    Returns:
        List of result dicts with: start_frame, end_frame, predicted,
        confidence, expected (if provided), correct (if expected provided).
    """
    video_path = Path(video_path)
    keypoint_dir = Path(keypoint_dir)
    vid_id = video_path.stem

    # Extract keypoints if needed (for crop model)
    kps_full = None
    if config.get("input_type") == "crop":
        kp_path = keypoint_dir / f"{vid_id}.npy"
        if not kp_path.exists():
            logger.info("Extracting keypoints for %s...", vid_id)
            _get_extract_keypoints()(
                [{"video_id": vid_id, "video_path": str(video_path)}],
                keypoint_dir,
            )
        if kp_path.exists():
            kps_full = np.load(kp_path)

    results = []
    for rep in reps:
        start = rep["start_frame"]
        end = rep["end_frame"]

        # Slice keypoints for this rep
        kps_rep = None
        if kps_full is not None:
            kps_rep = kps_full[start:min(end + 1, len(kps_full))]

        label, conf = _classify_rep(
            model, config, str(video_path), start, end, kps_rep, device,
        )

        result = {
            "start_frame": start,
            "end_frame": end,
            "predicted": label,
            "confidence": conf,
        }

        if "expected" in rep:
            result["expected"] = rep["expected"].upper()
            result["correct"] = label == result["expected"]

        results.append(result)

    return results


# ============================================================
# Approach 2: Automatic rep detection via state machine
# ============================================================


def infer_automatic(
    video_path: str | Path,
    model: PushUpR3D,
    config: dict,
    keypoint_dir: str | Path = "keypoints",
    device: str = "cpu",
    down_threshold: float = 90.0,
    up_threshold: float = 160.0,
    min_rep_frames: int = 10,
) -> list[dict]:
    """Automatically detect and classify reps from a video.

    Pipeline: YOLO keypoints -> state machine -> R3D classification.

    Args:
        video_path: Path to the video file.
        model: Loaded R3D model.
        config: Model config from checkpoint.
        keypoint_dir: Directory for YOLO keypoint .npy files.
        device: Torch device string.
        down_threshold: Elbow angle for "down" position.
        up_threshold: Elbow angle for "up" position.
        min_rep_frames: Minimum frames for a valid rep (filters noise).

    Returns:
        List of result dicts with: rep_number, start_frame, end_frame,
        predicted, confidence.
    """
    video_path = Path(video_path)
    keypoint_dir = Path(keypoint_dir)
    vid_id = video_path.stem

    # Step 1: Extract YOLO keypoints
    kp_path = keypoint_dir / f"{vid_id}.npy"
    if not kp_path.exists():
        logger.info("Extracting keypoints for %s...", vid_id)
        extract_keypoints(
            [{"video_id": vid_id, "video_path": str(video_path)}],
            keypoint_dir,
        )

    if not kp_path.exists():
        logger.error("Keypoint extraction failed for %s", vid_id)
        return []

    kps_full = np.load(kp_path)  # (T, 12, 3)

    # Step 2: Run state machine to detect rep boundaries
    sm = PushUpStateMachine(
        down_threshold=down_threshold,
        up_threshold=up_threshold,
    )
    boundaries = sm.segment_sequence(kps_full)

    # Filter out degenerate reps
    boundaries = [(s, e) for s, e in boundaries if e - s >= min_rep_frames]

    if not boundaries:
        logger.warning("No reps detected in %s", vid_id)
        return []

    logger.info("Detected %d reps in %s", len(boundaries), vid_id)

    # Step 3: Classify each detected rep
    results = []
    for i, (start, end) in enumerate(boundaries):
        kps_rep = kps_full[start:min(end + 1, len(kps_full))]

        label, conf = _classify_rep(
            model, config, str(video_path), start, end, kps_rep, device,
        )

        results.append({
            "rep_number": i + 1,
            "start_frame": start,
            "end_frame": end,
            "predicted": label,
            "confidence": conf,
        })

    return results


# ============================================================
# Summary helpers
# ============================================================


def summarize(results: list[dict]) -> dict:
    """Summarize inference results.

    Returns:
        Dict with: total_reps, good_reps, bad_reps, and optionally
        accuracy (if ground truth was provided).
    """
    total = len(results)
    good = sum(1 for r in results if r["predicted"] == "GOOD")
    bad = total - good

    summary = {
        "total_reps": total,
        "good_reps": good,
        "bad_reps": bad,
    }

    # If ground truth available
    if any("correct" in r for r in results):
        correct = sum(1 for r in results if r.get("correct", False))
        tested = sum(1 for r in results if "correct" in r)
        summary["accuracy"] = correct / tested if tested > 0 else 0.0
        summary["correct"] = correct
        summary["tested"] = tested

    return summary


def print_results(results: list[dict], title: str = "Inference Results") -> None:
    """Pretty-print inference results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    if not results:
        print("  No reps detected.")
        return

    has_expected = any("expected" in r for r in results)
    has_rep_num = any("rep_number" in r for r in results)

    # Header
    header = ""
    if has_rep_num:
        header += f"{'Rep':<5}"
    header += f"{'Frames':<15} {'Predicted':<10} {'Conf':<8}"
    if has_expected:
        header += f"{'Expected':<10} {'Match':<6}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for r in results:
        line = ""
        if has_rep_num:
            line += f"{r.get('rep_number', '-'):<5}"
        line += f"{r['start_frame']:>4}-{r['end_frame']:<8} {r['predicted']:<10} {r['confidence']:<8.1%}"
        if has_expected:
            match = "Y" if r.get("correct") else "N"
            line += f"{r.get('expected', '?'):<10} {match:<6}"
        print(f"  {line}")

    s = summarize(results)
    print(f"\n  Total: {s['total_reps']} reps — {s['good_reps']} GOOD, {s['bad_reps']} BAD")
    if "accuracy" in s:
        print(f"  Accuracy: {s['correct']}/{s['tested']} = {s['accuracy']:.1%}")


# ============================================================
# Approach 3: Live webcam demo
# ============================================================

# YOLO COCO-17 to unified 12-joint mapping
_YOLO_TO_UNIFIED = {
    5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5,
    11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11,
}

# Skeleton connections for drawing (unified indices)
_SKELETON_PAIRS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11),
]


def _yolo_to_unified(native_kps: np.ndarray) -> np.ndarray:
    """Map YOLO 17-joint keypoints to unified 12-joint format."""
    unified = np.zeros((12, 3), dtype=np.float32)
    for native_idx, unified_idx in _YOLO_TO_UNIFIED.items():
        if native_idx < len(native_kps):
            unified[unified_idx] = native_kps[native_idx]
    return unified


def _classify_buffered_frames(
    model: PushUpR3D,
    config: dict,
    frames: list[np.ndarray],
    keypoints: list[np.ndarray],
    device: str,
) -> tuple[str, float]:
    """Classify a rep from buffered BGR frames + keypoints.

    Uniformly samples n_frames from the buffer, preprocesses, and runs R3D.
    """
    n_frames = config.get("n_frames", 16)
    input_type = config.get("input_type", "full")
    total = len(frames)

    if total == 0:
        return "BAD", 0.5

    # Uniformly sample indices from the buffer
    indices = np.linspace(0, total - 1, n_frames).astype(int)

    if input_type == "crop" and keypoints:
        processed = []
        for i in indices:
            frame = frames[min(i, total - 1)]
            kp = keypoints[min(i, len(keypoints) - 1)]
            processed.append(_preprocess_cropped_frame(frame, kp))
    else:
        processed = [_preprocess_full_frame(frames[min(i, total - 1)]) for i in indices]

    tensor = torch.from_numpy(_to_tensor(processed)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        confidence = probs[0, pred].item()

    return "GOOD" if pred == 0 else "BAD", confidence


def _draw_skeleton(frame: np.ndarray, unified_kps: np.ndarray, min_conf: float = 0.3) -> None:
    """Draw skeleton overlay on frame (in-place)."""
    for i, j in _SKELETON_PAIRS:
        if unified_kps[i, 2] > min_conf and unified_kps[j, 2] > min_conf:
            pt1 = (int(unified_kps[i, 0]), int(unified_kps[i, 1]))
            pt2 = (int(unified_kps[j, 0]), int(unified_kps[j, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for k in range(12):
        if unified_kps[k, 2] > min_conf:
            pt = (int(unified_kps[k, 0]), int(unified_kps[k, 1]))
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def live_demo(
    model: PushUpR3D,
    config: dict,
    device: str = "cpu",
    yolo_model_path: str = "yolo11s-pose.pt",
    camera_index: int = 0,
    down_threshold: float = 90.0,
    up_threshold: float = 160.0,
    min_rep_frames: int = 10,
    show_skeleton: bool = True,
) -> list[dict]:
    """Run live webcam demo with real-time rep counting + form classification.

    Pipeline per frame:
        1. YOLO pose → keypoints
        2. State machine → detect rep boundaries
        3. On rep completion → R3D classifies the buffered frames

    Controls:
        Q / ESC — quit
        R       — reset counters

    Args:
        model: Loaded R3D model.
        config: Model config from checkpoint.
        device: Torch device string.
        yolo_model_path: Path to YOLO pose model weights.
        camera_index: Webcam index (0 = default camera).
        down_threshold: Elbow angle for "down" position.
        up_threshold: Elbow angle for "up" position.
        min_rep_frames: Minimum frames for a valid rep.
        show_skeleton: Draw keypoint skeleton overlay.

    Returns:
        List of result dicts for all completed reps.
    """
    from ultralytics import YOLO

    yolo = YOLO(yolo_model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sm = PushUpStateMachine(
        down_threshold=down_threshold,
        up_threshold=up_threshold,
    )

    # Buffers for current rep's frames and keypoints
    rep_frames: list[np.ndarray] = []
    rep_keypoints: list[np.ndarray] = []
    in_rep = False

    # Results
    all_results: list[dict] = []
    good_count = 0
    bad_count = 0
    last_label = ""
    last_conf = 0.0

    # Colors
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 220)

    print("Live Demo Started")
    print("  Controls: Q/ESC = quit, R = reset counters")
    print(f"  State machine thresholds: down={down_threshold}, up={up_threshold}")
    print(f"  Model: {config.get('unfreeze', 'frozen')}, input={config.get('input_type', 'full')}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # Run YOLO pose estimation
        results = yolo(frame, device=device, conf=0.5, verbose=False)
        result = results[0]

        unified_kps = np.zeros((12, 3), dtype=np.float32)
        elbow_angle = 0.0

        if result.keypoints is not None and len(result.keypoints):
            kps_data = result.keypoints.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(len(kps_data))
            best_idx = int(np.argmax(scores))
            native_kps = kps_data[best_idx]
            unified_kps = _yolo_to_unified(native_kps)
            elbow_angle = compute_elbow_angle(unified_kps)

            if show_skeleton:
                _draw_skeleton(display, unified_kps)

        # Update state machine
        prev_state = sm.state
        sm.update(elbow_angle)
        curr_state = sm.state

        # Track rep buffering
        from state_machine import PushUpPhase

        if prev_state == PushUpPhase.UP and curr_state == PushUpPhase.GOING_DOWN:
            # Rep starting — begin buffering
            in_rep = True
            rep_frames = []
            rep_keypoints = []

        if in_rep:
            rep_frames.append(frame.copy())
            rep_keypoints.append(unified_kps.copy())

        # Rep completed
        if prev_state == PushUpPhase.GOING_UP and curr_state == PushUpPhase.UP:
            in_rep = False
            if len(rep_frames) >= min_rep_frames:
                label, conf = _classify_buffered_frames(
                    model, config, rep_frames, rep_keypoints, device,
                )
                last_label = label
                last_conf = conf

                if label == "GOOD":
                    good_count += 1
                else:
                    bad_count += 1

                all_results.append({
                    "rep_number": good_count + bad_count,
                    "predicted": label,
                    "confidence": conf,
                    "n_frames": len(rep_frames),
                })
                print(f"  Rep {good_count + bad_count}: {label} ({conf:.1%})")

            rep_frames = []
            rep_keypoints = []

        # ---- Draw HUD overlay ----
        # Background panel
        cv2.rectangle(display, (5, 5), (350, 175), BLACK, -1)
        cv2.rectangle(display, (5, 5), (350, 175), WHITE, 2)

        # Rep counts
        cv2.putText(display, f"GOOD: {good_count}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
        cv2.putText(display, f"BAD:  {bad_count}", (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
        cv2.putText(display, f"Total: {good_count + bad_count}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # Current state
        state_str = curr_state.name.replace("_", " ")
        cv2.putText(display, f"Phase: {state_str}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.putText(display, f"Elbow: {elbow_angle:.0f} deg", (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

        # Last rep result
        if last_label:
            color = GREEN if last_label == "GOOD" else RED
            cv2.putText(display, f"Last: {last_label} ({last_conf:.0%})", (15, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Push-Up Live Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key == ord("r"):  # Reset
            sm.reset()
            good_count = 0
            bad_count = 0
            last_label = ""
            last_conf = 0.0
            all_results.clear()
            in_rep = False
            rep_frames = []
            rep_keypoints = []
            print("  Counters reset.")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nSession ended: {good_count} GOOD, {bad_count} BAD, {good_count + bad_count} total")
    return all_results
