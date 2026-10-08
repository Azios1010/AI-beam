"""Detector behavior checked without loading weights or installing YOLO."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aura_beam import EnsembleDetector


def detection(box: list[float], confidence: float, cls_id: int = 0) -> dict:
    return {"box": np.array(box, dtype=float), "conf": confidence, "cls": cls_id}


def test_weighted_fusion_combines_matching_boxes() -> None:
    detector = EnsembleDetector.__new__(EnsembleDetector)
    detector.fuse_iou_threshold = 0.35
    primary = detection([0, 0, 10, 10], 0.9)
    secondary = detection([2, 0, 12, 10], 0.6)
    fused = detector._fuse_model_detections_weighted([primary], [secondary])

    assert len(fused) == 1
    np.testing.assert_allclose(fused[0]["box"], [0.8, 0, 10.8, 10])
    assert fused[0]["conf"] == 0.9


def test_fusion_keeps_different_classes_separate() -> None:
    detector = EnsembleDetector.__new__(EnsembleDetector)
    detector.fuse_iou_threshold = 0.35
    primary = detection([0, 0, 10, 10], 0.9, cls_id=0)
    secondary = detection([0, 0, 10, 10], 0.6, cls_id=1)

    assert len(detector._fuse_model_detections_weighted([primary], [secondary])) == 2
    assert len(detector._fuse_model_detections_nms([primary], [secondary])) == 2


def test_nms_retains_highest_confidence_box() -> None:
    detector = EnsembleDetector.__new__(EnsembleDetector)
    detector.fuse_iou_threshold = 0.35
    fused = detector._fuse_model_detections_nms(
        [detection([0, 0, 10, 10], 0.9)], [detection([1, 0, 11, 10], 0.6)],
    )
    assert len(fused) == 1
    assert fused[0]["conf"] == 0.9


def test_detector_rejects_missing_explicit_path() -> None:
    with pytest.raises(ValueError, match="primary_model_path is required"):
        EnsembleDetector(detector_mode="primary")


def test_detector_checks_files_before_importing_optional_dependency(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing model weights file"):
        EnsembleDetector(primary_model_path=str(tmp_path / "absent.pt"), detector_mode="primary")


def test_detector_rejects_directory_as_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        EnsembleDetector(primary_model_path=str(tmp_path), detector_mode="primary")


def test_detector_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unsupported detector mode"):
        EnsembleDetector(detector_mode="invalid")


@pytest.mark.parametrize("mode", ["primary", "secondary", "ensemble_nms", "ensemble_weighted"])
def test_only_models_required_by_selected_mode_are_loaded(tmp_path, monkeypatch, mode: str) -> None:
    primary = tmp_path / "primary.pt"
    secondary = tmp_path / "secondary.pt"
    primary.write_bytes(b"stub")
    secondary.write_bytes(b"stub")
    loaded = []

    def fake_yolo(path: str) -> object:
        loaded.append(path)
        return object()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    EnsembleDetector(
        primary_model_path=str(primary) if mode != "secondary" else None,
        secondary_model_path=str(secondary) if mode != "primary" else None,
        detector_mode=mode,
    )
    expected = ([str(primary)] if mode != "secondary" else []) + ([str(secondary)] if mode != "primary" else [])
    assert loaded == expected


def test_optional_dependency_failure_has_install_instruction(tmp_path, monkeypatch) -> None:
    weights = tmp_path / "input.pt"
    weights.write_bytes(b"stub")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ModuleNotFoundError, match="detector extra"):
        EnsembleDetector(primary_model_path=str(weights), detector_mode="primary")
