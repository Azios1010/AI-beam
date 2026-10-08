"""Tests for greedy pixel-distance matching."""
import pytest

from aura_beam import DetectionInput, TrackPrediction, associate_detections_by_distance


def test_association_respects_class_and_distance() -> None:
    tracks = [TrackPrediction(1, 0, (10, 10)), TrackPrediction(2, 1, (11, 10))]
    detections = [DetectionInput(0, 0, (12, 10)), DetectionInput(1, 0, (100, 100))]
    matches = associate_detections_by_distance(tracks, detections, max_distance=5)
    assert len(matches) == 1
    assert matches[0].track_id == 1
    assert matches[0].detection_index == 0
    assert matches[0].cost == pytest.approx(2)


def test_association_matches_each_track_and_detection_once() -> None:
    tracks = [TrackPrediction(1, 0, (0, 0)), TrackPrediction(2, 0, (10, 0))]
    detections = [DetectionInput(0, 0, (1, 0)), DetectionInput(1, 0, (2, 0))]
    matches = associate_detections_by_distance(tracks, detections, max_distance=20)
    assert [(m.track_id, m.detection_index) for m in matches] == [(1, 0), (2, 1)]


def test_association_handles_empty_inputs() -> None:
    assert associate_detections_by_distance([], [], max_distance=5) == []


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_association_rejects_invalid_distance(distance: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        associate_detections_by_distance([], [], max_distance=distance)
