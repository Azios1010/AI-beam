"""Direct image bounding-box projection onto a rectangular grid."""
from dataclasses import dataclass
import math

from .grid_geometry import GridRectangle, merge_rectangles


@dataclass(frozen=True)
class ControlTarget:
    """An image bounding box specified by center, width and height in pixels."""

    x: float
    y: float
    box_width: float
    box_height: float


class MatrixGridController:
    """Map each bounding box to all grid cells it overlaps."""

    def __init__(self, matrix_width: int = 8, matrix_height: int = 8) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (matrix_width, matrix_height)):
            raise ValueError("grid dimensions must be positive integers")
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height

    def get_dark_rectangles(
        self,
        targets: list[ControlTarget],
        screen_width: float,
        screen_height: float,
    ) -> list[GridRectangle]:
        """Clip boxes to the image, project their extent, then merge regions."""
        if any(not math.isfinite(v) or v <= 0 for v in (screen_width, screen_height)):
            raise ValueError("screen dimensions must be finite and positive")
        rectangles = []
        for target in targets:
            if not all(math.isfinite(v) for v in (target.x, target.y, target.box_width, target.box_height)):
                raise ValueError("bounding box coordinates must be finite")
            if target.box_width <= 0 or target.box_height <= 0:
                raise ValueError("bounding box dimensions must be positive")
            left = max(0.0, target.x - target.box_width / 2.0)
            right = min(screen_width, target.x + target.box_width / 2.0)
            top = max(0.0, target.y - target.box_height / 2.0)
            bottom = min(screen_height, target.y + target.box_height / 2.0)
            if left >= right or top >= bottom:
                continue
            col_start = int(math.floor(left / screen_width * self.matrix_width))
            col_end = min(self.matrix_width - 1, int(math.ceil(right / screen_width * self.matrix_width)) - 1)
            row_start = int(math.floor(top / screen_height * self.matrix_height))
            row_end = min(self.matrix_height - 1, int(math.ceil(bottom / screen_height * self.matrix_height)) - 1)
            rectangles.append(GridRectangle(col_start, col_end, row_start, row_end))
        return merge_rectangles(rectangles)
