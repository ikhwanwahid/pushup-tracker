"""Helper tool for annotating push-up rep boundaries.

Opens a matplotlib window to step through video frames with frame numbers.
Use this to find start_frame and end_frame for each rep.

Controls (click on the figure window first):
    RIGHT / D      — next frame
    LEFT  / A      — previous frame
    UP    / F      — jump forward 10 frames
    DOWN  / B      — jump backward 10 frames
    Q              — quit current video, move to next
    ESCAPE         — quit entirely

Usage:
    python annotate_helper.py video1.mp4 video2.mp4 video3.mp4
    python annotate_helper.py /path/to/video/folder/
    python annotate_helper.py *.mp4
"""

import glob
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def play_with_frame_numbers(video_path: str) -> bool:
    """Returns True to continue to next video, False to quit entirely."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  SKIP: Cannot open {video_path}")
        return True

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Pre-load all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    total = len(frames)

    if total == 0:
        print(f"  SKIP: No frames in {video_path}")
        return True

    print(f"  Loaded {total} frames, {fps:.0f}fps, {total/fps:.1f}s")

    state = {"current": 0, "quit_all": False}

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    def show_frame():
        ax.clear()
        ax.imshow(frames[state["current"]])
        ax.set_title(
            f"{Path(video_path).name}  |  "
            f"Frame: {state['current']} / {total - 1}  |  "
            f"Time: {state['current']/fps:.2f}s",
            fontsize=13, fontweight="bold",
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["current"] = min(state["current"] + 1, total - 1)
        elif event.key in ("left", "a"):
            state["current"] = max(state["current"] - 1, 0)
        elif event.key in ("up", "f"):
            state["current"] = min(state["current"] + 10, total - 1)
        elif event.key in ("down", "b"):
            state["current"] = max(state["current"] - 10, 0)
        elif event.key == "q":
            plt.close(fig)
            return
        elif event.key == "escape":
            state["quit_all"] = True
            plt.close(fig)
            return
        show_frame()
        print(f"  Frame {state['current']}", end="\r")

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_frame()
    plt.show()

    return not state["quit_all"]


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python annotate_helper.py video1.mp4 video2.mp4 ...")
        print("  python annotate_helper.py /path/to/video/folder/")
        sys.exit(1)

    # Collect all video paths
    video_paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV", "*.MP4"):
                video_paths.extend(sorted(p.glob(ext)))
        elif p.exists():
            video_paths.append(p)
        else:
            # Try glob expansion
            expanded = sorted(glob.glob(arg))
            video_paths.extend(Path(x) for x in expanded)

    video_paths = [p for p in video_paths if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]

    if not video_paths:
        print("No video files found.")
        sys.exit(1)

    print(f"Found {len(video_paths)} videos")
    print("Controls: RIGHT/D=next, LEFT/A=prev, UP/F=+10, DOWN/B=-10")
    print("          Q=next video, ESC=quit all")
    print()

    for i, vpath in enumerate(video_paths):
        print(f"[{i+1}/{len(video_paths)}] {vpath.name}")
        keep_going = play_with_frame_numbers(str(vpath))
        if not keep_going:
            print("\nStopped early.")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
