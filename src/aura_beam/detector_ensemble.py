"""YOLO detection wrappers, class-aware box fusion and IoU helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO


IOU_FUSE_THRESHOLD = 0.35


class EnsembleDetector:
    """Detector wrapper supporting primary, secondary, and ensemble modes."""

    def __init__(
        self,
        primary_model_path: str | None = None,
        secondary_model_path: str | None = None,
        conf_threshold: float = 0.25,
        detector_mode: str = "primary",
        fuse_iou_threshold: float = IOU_FUSE_THRESHOLD,
    ) -> None:
        self.primary_model_path = str(primary_model_path) if primary_model_path is not None else None
        self.secondary_model_path = str(secondary_model_path) if secondary_model_path is not None else None
        self.conf_threshold = float(conf_threshold)
        self.detector_mode = str(detector_mode)
        self.fuse_iou_threshold = float(fuse_iou_threshold)

        valid_modes = {"primary", "secondary", "ensemble_nms", "ensemble_weighted"}
        if self.detector_mode not in valid_modes:
            raise ValueError(f"unsupported detector mode: {self.detector_mode}")

        resolved_primary = None
        resolved_secondary = None
        if self.detector_mode != "secondary":
            if self.primary_model_path is None:
                raise ValueError("primary_model_path is required for this detector mode")
            resolved_primary = self._resolve_model_path(self.primary_model_path)
        if self.detector_mode != "primary":
            if self.secondary_model_path is None:
                raise ValueError("secondary_model_path is required for this detector mode")
            resolved_secondary = self._resolve_model_path(self.secondary_model_path)

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            if exc.name != "ultralytics":
                raise
            raise ModuleNotFoundError(
                'YOLO inference requires the detector extra: pip install -e ".[detector]"'
            ) from exc

        self.primary_model = YOLO(resolved_primary) if resolved_primary is not None else None
        self.secondary_model = YOLO(resolved_secondary) if resolved_secondary is not None else None

    def detect(self, frame) -> list[dict]:
        """
        Chạy hai model trên cùng một frame rồi hợp nhất detection.
        """
        if self.detector_mode == "primary":
            return self._run_model_detection(self.primary_model, frame, self.primary_model_path)
        if self.detector_mode == "secondary":
            return self._run_model_detection(self.secondary_model, frame, self.secondary_model_path)

        primary_detections = self._run_model_detection(self.primary_model, frame, self.primary_model_path)
        secondary_detections = self._run_model_detection(self.secondary_model, frame, self.secondary_model_path)
        if self.detector_mode == "ensemble_nms":
            return self._fuse_model_detections_nms(primary_detections, secondary_detections)
        return self._fuse_model_detections_weighted(primary_detections, secondary_detections)


    @staticmethod
    def _resolve_model_path(path_str: str) -> str:
        """
        Bắt buộc file model phải tồn tại.
        """
        if not Path(path_str).is_file():
            raise FileNotFoundError(f"Missing model weights file: {path_str}")
        return path_str

    def _run_model_detection(self, model: YOLO, frame, model_name: str) -> list[dict]:
        """
        Chạy một model YOLO ở chế độ detect để lấy bbox thô.
        """
        detections: list[dict] = []
        results = model.predict(frame, conf=self.conf_threshold, verbose=False)

        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4), dtype=np.float32)
        cls = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []

        for box, cls_id, conf in zip(xyxy, cls, confs):
            detections.append(
                {
                    "box": box.astype(np.float64),
                    "cls": int(cls_id),
                    "conf": float(conf),
                    "source": model_name,
                }
            )

        return detections

    def _fuse_model_detections_weighted(self, primary_dets: list[dict], secondary_dets: list[dict]) -> list[dict]:
        """
        Hợp nhất detection của hai model bằng weighted box fusion đơn giản.
        """
        fused_detections: list[dict] = []
        used_secondary: set[int] = set()

        for primary in primary_dets:
            best_match_index = -1
            best_iou = 0.0

            for sec_index, secondary in enumerate(secondary_dets):
                if sec_index in used_secondary or primary["cls"] != secondary["cls"]:
                    continue

                iou = self.compute_iou(primary["box"], secondary["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_index = sec_index

            if best_match_index >= 0 and best_iou >= self.fuse_iou_threshold:
                secondary = secondary_dets[best_match_index]
                used_secondary.add(best_match_index)

                weight_primary = max(primary["conf"], 1e-6)
                weight_secondary = max(secondary["conf"], 1e-6)
                fused_box = (
                    primary["box"] * weight_primary + secondary["box"] * weight_secondary
                ) / (weight_primary + weight_secondary)

                fused_detections.append(
                    {
                        "box": fused_box,
                        "cls": primary["cls"],
                        "conf": max(primary["conf"], secondary["conf"]),
                        "source": "ensemble",
                    }
                )
            else:
                fused_detections.append(primary)

        for sec_index, secondary in enumerate(secondary_dets):
            if sec_index not in used_secondary:
                fused_detections.append(secondary)

        fused_detections.sort(key=lambda item: item["conf"], reverse=True)
        return fused_detections

    def _fuse_model_detections_nms(self, primary_dets: list[dict], secondary_dets: list[dict]) -> list[dict]:
        candidates = [dict(item) for item in primary_dets] + [dict(item) for item in secondary_dets]
        candidates.sort(key=lambda item: item["conf"], reverse=True)

        kept: list[dict] = []
        for candidate in candidates:
            suppress = False
            for kept_item in kept:
                if candidate["cls"] != kept_item["cls"]:
                    continue
                if self.compute_iou(candidate["box"], kept_item["box"]) >= self.fuse_iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)

        return kept

    @staticmethod
    def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Tính Intersection over Union để đo mức chồng lấp giữa 2 bbox.
        """
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    @staticmethod
    def center_of_box(box: np.ndarray) -> tuple[float, float]:
        """
        Tính tâm hình học của bounding box.
        """
        return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)
