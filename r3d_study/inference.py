"""Inference pipelines for push-up rep counting + form classification.

Two modes:
  1. infer_automatic() — state machine auto-detects reps from recorded video
  2. live_record_and_classify() — webcam record, then offline processing

Both use the best saved R3D model and YOLO keypoints.
"""

import logging
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
# Automatic rep detection via state machine
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
        extract_keypoints = _get_extract_keypoints()
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
    # Pad boundaries to give classifier full rep context
    pad = 10
    results = []
    for i, (start, end) in enumerate(boundaries):
        s = max(0, start - pad)
        e = min(len(kps_full) - 1, end + pad)
        kps_rep = kps_full[s:e + 1]

        label, conf = _classify_rep(
            model, config, str(video_path), s, e, kps_rep, device,
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
    has_frames = any("start_frame" in r for r in results)

    # Header
    header = ""
    if has_rep_num:
        header += f"{'Rep':<5}"
    if has_frames:
        header += f"{'Frames':<15}"
    header += f"{'Predicted':<10} {'Conf':<8}"
    if has_expected:
        header += f"{'Expected':<10} {'Match':<6}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for r in results:
        line = ""
        if has_rep_num:
            line += f"{r.get('rep_number', '-'):<5}"
        if "start_frame" in r:
            line += f"{r['start_frame']:>4}-{r['end_frame']:<8} "
        line += f"{r['predicted']:<10} {r['confidence']:<8.1%}"
        if has_expected:
            match = "Y" if r.get("correct") else "N"
            line += f"{r.get('expected', '?'):<10} {match:<6}"
        print(f"  {line}")

    s = summarize(results)
    print(f"\n  Total: {s['total_reps']} reps — {s['good_reps']} GOOD, {s['bad_reps']} BAD")
    if "accuracy" in s:
        print(f"  Accuracy: {s['correct']}/{s['tested']} = {s['accuracy']:.1%}")


# ============================================================
# Live webcam: Record then classify
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


def live_record_and_classify(
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
    """Record webcam video, then process offline for accurate classification.

    Pipeline:
        1. Live preview with skeleton overlay
        2. Press S to start/stop recording
        3. On stop: YOLO extracts keypoints over full recording
        4. State machine detects rep boundaries
        5. R3D classifies each rep with proper crops

    Controls:
        S       — start/stop recording
        Q / ESC — quit

    Returns:
        List of result dicts for all classified reps.
    """
    from ultralytics import YOLO

    yolo = YOLO(yolo_model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    recording = False
    recorded_frames: list[np.ndarray] = []
    all_results: list[dict] = []
    status_msg = "Press S to start recording"
    result_msg = ""
    good_count = 0
    bad_count = 0

    # Colors
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 220)
    ORANGE = (0, 140, 255)

    print("Record & Classify Demo")
    print("  Controls: S = start/stop recording, Q/ESC = quit")
    print(f"  Model: {config.get('unfreeze', 'frozen')}, input={config.get('input_type', 'full')}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # Live YOLO skeleton preview
        if show_skeleton:
            results = yolo(frame, device=device, conf=0.5, verbose=False)
            result = results[0]
            if result.keypoints is not None and len(result.keypoints):
                kps_data = result.keypoints.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(len(kps_data))
                best_idx = int(np.argmax(scores))
                unified_kps = _yolo_to_unified(kps_data[best_idx])
                _draw_skeleton(display, unified_kps)

        if recording:
            recorded_frames.append(frame.copy())

        # ---- HUD ----
        cv2.rectangle(display, (5, 5), (400, 140), BLACK, -1)
        cv2.rectangle(display, (5, 5), (400, 140), WHITE, 2)

        if recording:
            cv2.putText(display, f"RECORDING  [{len(recorded_frames)} frames]", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            # Blinking red dot
            if len(recorded_frames) % 30 < 15:
                cv2.circle(display, (380, 28), 10, RED, -1)
        else:
            cv2.putText(display, status_msg, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        if good_count + bad_count > 0:
            cv2.putText(display, f"GOOD: {good_count}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            cv2.putText(display, f"BAD: {bad_count}", (200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        if result_msg:
            cv2.putText(display, result_msg, (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

        cv2.putText(display, "S=record  Q=quit", (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.imshow("Record & Classify", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            if not recording:
                # Start recording
                recording = True
                recorded_frames = []
                status_msg = "Recording..."
                result_msg = ""
                print("  Recording started...")
            else:
                # Stop recording — process the video
                recording = False
                n_frames_rec = len(recorded_frames)
                print(f"  Recording stopped. {n_frames_rec} frames captured.")

                if n_frames_rec < 20:
                    status_msg = "Too short! Press S to try again"
                    result_msg = ""
                    continue

                status_msg = "Processing..."
                result_msg = "Extracting keypoints + classifying..."
                # Force display update
                cv2.putText(display, "Processing...", (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, ORANGE, 3)
                cv2.imshow("Record & Classify", display)
                cv2.waitKey(1)

                # -- Step 1: YOLO keypoints on all recorded frames --
                print("  Extracting keypoints...")
                all_kps = []
                for i, f in enumerate(recorded_frames):
                    res = yolo(f, device=device, conf=0.5, verbose=False)[0]
                    unified = np.zeros((12, 3), dtype=np.float32)
                    if res.keypoints is not None and len(res.keypoints):
                        kd = res.keypoints.data.cpu().numpy()
                        sc = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(len(kd))
                        nkp = kd[int(np.argmax(sc))]
                        unified = _yolo_to_unified(nkp)
                    all_kps.append(unified)
                    if (i + 1) % 50 == 0:
                        print(f"    {i+1}/{n_frames_rec} frames...")

                kps_array = np.stack(all_kps)

                # -- Step 2: State machine to detect reps --
                sm = PushUpStateMachine(
                    down_threshold=down_threshold,
                    up_threshold=up_threshold,
                )
                boundaries = sm.segment_sequence(kps_array)
                boundaries = [(s, e) for s, e in boundaries if e - s >= min_rep_frames]
                print(f"  Detected {len(boundaries)} reps")

                if not boundaries:
                    status_msg = f"No reps detected. Press S to retry"
                    result_msg = ""
                    continue

                # -- Step 3: Classify each rep --
                session_results = []
                for i, (start, end) in enumerate(boundaries):
                    rep_f = recorded_frames[start:end + 1]
                    rep_kps = all_kps[start:end + 1]

                    n_sample = config.get("n_frames", 16)
                    sample_idx = np.linspace(0, len(rep_f) - 1, n_sample).astype(int)

                    processed = []
                    for si in sample_idx:
                        processed.append(
                            _preprocess_cropped_frame(rep_f[si], rep_kps[si])
                        )

                    tensor = torch.from_numpy(_to_tensor(processed)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = F.softmax(logits, dim=-1)
                        pred = logits.argmax(dim=-1).item()
                        conf = probs[0, pred].item()

                    label = "GOOD" if pred == 0 else "BAD"
                    if label == "GOOD":
                        good_count += 1
                    else:
                        bad_count += 1

                    session_results.append({
                        "rep_number": good_count + bad_count,
                        "predicted": label,
                        "confidence": conf,
                        "start_frame": start,
                        "end_frame": end,
                    })
                    print(f"    Rep {i+1}: {label} ({conf:.1%}) [frames {start}-{end}]")

                all_results.extend(session_results)
                status_msg = f"Done! {good_count} GOOD, {bad_count} BAD. Press S for more"
                result_msg = f"Last session: {len(session_results)} reps detected"

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal: {good_count} GOOD, {bad_count} BAD")
    return all_results
