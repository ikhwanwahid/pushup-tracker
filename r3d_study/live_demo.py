"""Standalone live demo script — run from terminal, not Jupyter.

Usage:
    python live_demo.py
    python live_demo.py --camera 1
    python live_demo.py --down 80 --up 150
"""

import argparse
import torch
from inference import load_model, live_demo, print_results, summarize


def main():
    parser = argparse.ArgumentParser(description="Live push-up form demo")
    parser.add_argument("--model", default="outputs/r3d_best.pt", help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--down", type=float, default=90.0, help="Down threshold (degrees)")
    parser.add_argument("--up", type=float, default=160.0, help="Up threshold (degrees)")
    parser.add_argument("--no-skeleton", action="store_true", help="Hide skeleton overlay")
    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")

    model, config = load_model(args.model, device=device)
    print(f"Input type: {config.get('input_type')}")
    print(f"CV accuracy: {config.get('accuracy', 0):.1%}")
    print()

    results = live_demo(
        model=model,
        config=config,
        device=device,
        camera_index=args.camera,
        down_threshold=args.down,
        up_threshold=args.up,
        show_skeleton=not args.no_skeleton,
    )

    if results:
        print_results(results, "Session Results")
        s = summarize(results)
        print(f"\nFinal: {s['good_reps']} GOOD / {s['total_reps']} total")


if __name__ == "__main__":
    main()
