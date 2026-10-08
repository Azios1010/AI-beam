"""Basic image-position tracking and bounding-box grid projection."""

from .detector_ensemble import EnsembleDetector
from .grid_geometry import GridRectangle, merge_rectangles, rectangles_to_mask
from .sensor_fusion import KalmanFilter2D
from .tracking import DetectionInput, TrackMatch, TrackPrediction, associate_detections_by_distance
from .zone_logic import ControlTarget, MatrixGridController

__all__ = [
    "ControlTarget", "DetectionInput", "EnsembleDetector", "GridRectangle",
    "KalmanFilter2D", "MatrixGridController", "TrackMatch", "TrackPrediction",
    "associate_detections_by_distance", "merge_rectangles", "rectangles_to_mask",
]
