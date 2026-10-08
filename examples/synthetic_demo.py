"""Filter generated image positions and project synthetic boxes onto a grid."""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from aura_beam import ControlTarget, KalmanFilter2D, MatrixGridController


def run_demo(frames: int, fps: float, seed: int) -> None:
    """Print a repeatable image-position example without input assets."""
    if frames <= 0 or fps <= 0 or not np.isfinite(fps):
        raise ValueError("frames and fps must be positive; fps must be finite")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    tracker = KalmanFilter2D(dt=1.0 / fps)
    tracker.initialize(x=320.0, y=180.0)
    controller = MatrixGridController()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["frame", "observed_x", "observed_y", "filtered_x", "filtered_y", "grid_rectangles"])
    for frame in range(frames):
        observed_x = 320.0 + frame * 2.0 + rng.normal(0, 2)
        observed_y = 180.0 + rng.normal(0, 2)
        x, y = tracker.update(observed_x, observed_y)
        target = ControlTarget(x=x, y=y, box_width=80.0, box_height=40.0)
        rectangles = controller.get_dark_rectangles([target], screen_width=640, screen_height=360)
        regions = ";".join(
            f"{r.col_start}:{r.col_end}:{r.row_start}:{r.row_end}" for r in rectangles
        )
        writer.writerow([
            frame, f"{observed_x:.2f}", f"{observed_y:.2f}",
            f"{x:.2f}", f"{y:.2f}", regions,
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        run_demo(args.frames, args.fps, args.seed)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
