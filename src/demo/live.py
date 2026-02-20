"""Live push-up tracking demo — webcam or video file.

Usage:
    # Webcam (default)
    uv run python -m src.demo.live

    # Video file
    uv run python -m src.demo.live --video path/to/pushup.mp4

    # Options
    uv run python -m src.demo.live --video path.mp4 --model mediapipe --save output.mp4
"""

import argparse
import time

import cv2
import numpy as np

from src.pose_estimation.base import PoseEstimatorBase
from src.pose_estimation.visualization import draw_skeleton
from src.counting.state_machine import PushUpStateMachine, PushUpPhase
from src.features.angles import compute_elbow_angle, compute_back_alignment
from src.quality.scoring import QualityScorer
from src.quality.feedback import generate_feedback


# Phase colors (BGR)
PHASE_COLORS = {
    PushUpPhase.UP: (0, 255, 0),         # green
    PushUpPhase.GOING_DOWN: (0, 255, 255),  # yellow
    PushUpPhase.DOWN: (0, 0, 255),        # red
    PushUpPhase.GOING_UP: (255, 165, 0),  # orange
}


def _draw_overlay(
    frame: np.ndarray,
    result,
    sm: PushUpStateMachine,
    phase: PushUpPhase,
    form_result: tuple[str, float, int] | None = None,
) -> None:
    """Draw the info overlay (rep count, phase, angles, timing, form) on the frame."""
    h, w = frame.shape[:2]

    elbow = compute_elbow_angle(result.unified_keypoints)
    back = compute_back_alignment(result.unified_keypoints)
    phase_color = PHASE_COLORS.get(phase, (0, 255, 0))

    # Semi-transparent background for text readability
    overlay = frame.copy()
    box_h = 207 if form_result else 175
    cv2.rectangle(overlay, (5, 5), (280, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Info lines
    lines = [
        (f"Reps: {sm.count}", (0, 255, 0), 0.8),
        (f"Phase: {phase.name}", phase_color, 0.6),
        (f"Elbow: {elbow:.0f} deg", (255, 255, 255), 0.6),
        (f"Back: {back:.0f} deg", (255, 255, 255), 0.6),
        (f"Inference: {result.inference_time_ms:.0f} ms", (180, 180, 180), 0.5),
    ]

    # Add form feedback line if available
    if form_result:
        form_label, form_conf, rep_num = form_result
        if form_label == "correct":
            form_color = (0, 255, 0)  # green
        else:
            form_color = (0, 0, 255)  # red
        lines.append(
            (f"Rep {rep_num}: {form_label.upper()} ({form_conf:.0%})", form_color, 0.6)
        )

    y = 30
    for text, color, scale in lines:
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 32

    # Phase indicator bar at the bottom
    bar_height = 8
    bar_y = h - bar_height - 5
    cv2.rectangle(frame, (5, bar_y), (w - 5, bar_y + bar_height), phase_color, -1)


def run_live_demo(
    estimator: PoseEstimatorBase,
    camera_id: int = 0,
    down_threshold: float = 90.0,
    up_threshold: float = 160.0,
) -> None:
    """Run a live webcam push-up tracking demo.

    Shows skeleton overlay, rep count, current phase, and elbow angle.
    Press 'q' to quit.

    Args:
        estimator: A PoseEstimatorBase instance.
        camera_id: Webcam device ID.
        down_threshold: State machine down threshold.
        up_threshold: State machine up threshold.
    """
    run_demo(estimator, source=camera_id, down_threshold=down_threshold, up_threshold=up_threshold, form_checker=None)


def run_demo(
    estimator: PoseEstimatorBase,
    source: int | str = 0,
    down_threshold: float = 90.0,
    up_threshold: float = 160.0,
    save_path: str | None = None,
    form_checker: "FormChecker | None" = None,
) -> None:
    """Run push-up tracking on webcam or video file.

    Args:
        estimator: A PoseEstimatorBase instance.
        source: Camera ID (int) for webcam, or file path (str) for video.
        down_threshold: State machine down threshold (degrees).
        up_threshold: State machine up threshold (degrees).
        save_path: If set, save annotated video to this path.
        form_checker: Optional FormChecker for real-time form feedback.
    """
    is_video = isinstance(source, str)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'video ' + source if is_video else 'camera ' + str(source)}")

    fps = cap.get(cv2.CAP_PROP_FPS) if is_video else 30.0
    frame_delay = max(1, int(1000 / fps)) if is_video else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    sm = PushUpStateMachine(down_threshold=down_threshold, up_threshold=up_threshold)
    writer = None

    source_label = source if is_video else f"Camera {source}"
    print(f"Starting demo: {estimator.model_name} | Source: {source_label}")
    if is_video and total_frames > 0:
        print(f"Video: {total_frames} frames @ {fps:.1f} FPS ({total_frames / fps:.1f}s)")
    print("Controls: [q] quit  [space] pause  [r] restart count")

    paused = False
    frame_num = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    # Video ended — hold on last frame
                    print(f"\nVideo finished. Final count: {sm.count} reps")
                    print("Press 'q' to close, 'r' to replay")
                    cv2.waitKey(0)
                break
            frame_num += 1

            result = estimator.predict_frame(frame)

            if result.detected:
                draw_skeleton(frame, result.unified_keypoints)
                phase = sm.update_from_keypoints(result.unified_keypoints)

                # Update form checker if available
                form_result = None
                if form_checker is not None:
                    form_result = form_checker.update(result.unified_keypoints, phase)

                _draw_overlay(frame, result, sm, phase, form_result=form_result)
            else:
                cv2.putText(frame, "No pose detected", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Progress bar for video files
            if is_video and total_frames > 0:
                h, w = frame.shape[:2]
                progress = frame_num / total_frames
                bar_w = int((w - 10) * progress)
                cv2.rectangle(frame, (5, h - 3), (5 + bar_w, h), (0, 200, 0), -1)

            # Save output
            if save_path and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            if writer is not None:
                writer.write(frame)

        cv2.imshow("Push-Up Tracker", frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            if paused:
                print("Paused — press space to resume")
        elif key == ord("r"):
            sm.reset()
            if is_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
            print("Reset — count restarted")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to: {save_path}")
    cv2.destroyAllWindows()
    print(f"Final count: {sm.count} reps")


# ---------------------------------------------------------------------------
# Model registry — add new models here as a single line
# Each entry: "cli_name" -> (module_path, class_name)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "yolo": ("src.pose_estimation.yolo_estimator", "YoloEstimator"),
    "mediapipe": ("src.pose_estimation.mediapipe_estimator", "MediaPipeEstimator"),
    # To add a new model, just add a line:
    # "movenet": ("src.pose_estimation.movenet_estimator", "MoveNetEstimator"),
    # "rtmpose": ("src.pose_estimation.rtmpose_estimator", "RTMPoseEstimator"),
}


def _load_model(name: str) -> "PoseEstimatorBase":
    """Lazily import and instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    module_path, class_name = MODEL_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def main():
    parser = argparse.ArgumentParser(description="Push-Up Tracker Demo")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (omit for webcam)")
    parser.add_argument("--model", type=str, default="yolo",
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Pose estimation model (default: yolo). "
                             f"Available: {', '.join(MODEL_REGISTRY.keys())}")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID for webcam mode (default: 0)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save annotated video to this path (e.g. output.mp4)")
    parser.add_argument("--down-threshold", type=float, default=90.0,
                        help="Elbow angle for 'down' position (default: 90)")
    parser.add_argument("--up-threshold", type=float, default=160.0,
                        help="Elbow angle for 'up' position (default: 160)")
    parser.add_argument("--form-model", type=str, default=None,
                        help="Path to trained ST-GCN model (.pt) for real-time form feedback")
    args = parser.parse_args()

    estimator = _load_model(args.model)

    # Load form checker if model path provided
    form_checker = None
    if args.form_model:
        from src.classification.form_checker import FormChecker
        form_checker = FormChecker(args.form_model)
        print(f"Form checker loaded: {args.form_model}")

    source = args.video if args.video else args.camera
    run_demo(
        estimator,
        source=source,
        down_threshold=args.down_threshold,
        up_threshold=args.up_threshold,
        save_path=args.save,
        form_checker=form_checker,
    )


if __name__ == "__main__":
    main()
