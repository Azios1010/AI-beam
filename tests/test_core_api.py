"""Basic Kalman filtering and standalone synthetic demo tests."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from aura_beam import KalmanFilter2D


def test_filter_predicts_constant_image_velocity() -> None:
    tracker = KalmanFilter2D(dt=1.0)
    tracker.initialize(10, 20)
    tracker.x[2, 0] = 2
    tracker.x[3, 0] = -1
    np.testing.assert_allclose(tracker.update(None, None), (12, 19))


def test_filter_corrects_position_toward_observation() -> None:
    tracker = KalmanFilter2D(dt=1.0)
    tracker.initialize(0, 0)
    x, y = tracker.update(10, 20)
    assert 0 < x < 10
    assert 0 < y < 20
    assert tracker.P[0, 0] < 1


def test_update_can_follow_explicit_prediction_without_predicting_twice() -> None:
    tracker = KalmanFilter2D(dt=1.0)
    tracker.initialize(0, 0)
    tracker.x[2, 0] = 2
    tracker.predict()
    assert tracker.update(None, None, predict_first=False) == (2, 0)


@pytest.mark.parametrize("kwargs", [{"dt": 0}, {"process_variance": -1}, {"measurement_variance": float("nan")}])
def test_filter_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        KalmanFilter2D(**kwargs)


def test_importing_core_does_not_load_optional_detector() -> None:
    code = "import sys; import aura_beam; assert 'ultralytics' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_synthetic_demo_is_repeatable_and_projects_regions() -> None:
    demo = Path(__file__).resolve().parents[1] / "examples" / "synthetic_demo.py"
    command = [sys.executable, str(demo), "--frames", "8", "--seed", "7"]
    first = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(first)))
    assert first == second
    assert len(rows) == 8
    assert all(row["grid_rectangles"] for row in rows)
    assert any(row["observed_x"] != row["filtered_x"] for row in rows)
