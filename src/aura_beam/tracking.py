"""Greedy class-aware association using image-plane Euclidean distance."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionInput:
    detection_index: int
    cls_id: int
    center: tuple[float, float]


@dataclass(frozen=True)
class TrackPrediction:
    track_id: int
    cls_id: int
    center: tuple[float, float]


@dataclass(frozen=True)
class TrackMatch:
    track_id: int
    detection_index: int
    cost: float


def associate_detections_by_distance(
    tracks: list[TrackPrediction],
    detections: list[DetectionInput],
    max_distance: float,
) -> list[TrackMatch]:
    """Match same-class pairs once each, in increasing pixel-distance order."""
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    candidates: list[tuple[float, int, int]] = []
    for track in tracks:
        for detection in detections:
            if track.cls_id != detection.cls_id:
                continue
            distance = float(np.hypot(
                detection.center[0] - track.center[0],
                detection.center[1] - track.center[1],
            ))
            if distance < max_distance:
                candidates.append((distance, track.track_id, detection.detection_index))
    candidates.sort(key=lambda item: item[0])
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[TrackMatch] = []
    for distance, track_id, detection_index in candidates:
        if track_id in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_id)
        used_detections.add(detection_index)
        matches.append(TrackMatch(track_id, detection_index, distance))
    return matches
