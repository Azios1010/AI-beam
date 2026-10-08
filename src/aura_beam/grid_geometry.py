"""Grid rectangles, boolean masks and region merging."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridRectangle:
    col_start: int
    col_end: int
    row_start: int
    row_end: int


def rectangles_to_mask(
    rectangles: list[GridRectangle],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("grid dimensions must be positive")
    mask = np.zeros((height, width), dtype=bool)
    for rect in rectangles:
        if not (
            0 <= rect.col_start <= rect.col_end < width
            and 0 <= rect.row_start <= rect.row_end < height
        ):
            raise ValueError(f"rectangle outside {width}x{height} grid: {rect}")
        mask[
            rect.row_start : rect.row_end + 1,
            rect.col_start : rect.col_end + 1,
        ] = True
    return mask


def merge_rectangles(rectangles: list[GridRectangle]) -> list[GridRectangle]:
    if not rectangles:
        return []

    def rects_overlap_or_adjacent(r1: GridRectangle, r2: GridRectangle) -> bool:
        col_overlap = max(0, min(r1.col_end, r2.col_end) - max(r1.col_start, r2.col_start) + 1)
        row_overlap = max(0, min(r1.row_end, r2.row_end) - max(r1.row_start, r2.row_start) + 1)

        # Edge adjacent counts as overlap for merging if the other dimension overlaps
        col_adjacent = (r1.col_end + 1 == r2.col_start) or (r2.col_end + 1 == r1.col_start)
        row_adjacent = (r1.row_end + 1 == r2.row_start) or (r2.row_end + 1 == r1.row_start)

        if col_overlap > 0 and row_overlap > 0:
            return True
        if col_overlap > 0 and row_adjacent:
            return True
        if row_overlap > 0 and col_adjacent:
            return True
        return False

    def merge_two(r1: GridRectangle, r2: GridRectangle) -> GridRectangle:
        return GridRectangle(
            min(r1.col_start, r2.col_start),
            max(r1.col_end, r2.col_end),
            min(r1.row_start, r2.row_start),
            max(r1.row_end, r2.row_end),
        )

    current_rects = list(rectangles)
    merged_any = True
    while merged_any:
        merged_any = False
        next_rects = []
        while current_rects:
            base = current_rects.pop(0)
            merged_this = False
            for i, other in enumerate(current_rects):
                if rects_overlap_or_adjacent(base, other):
                    merged_rect = merge_two(base, other)
                    current_rects.pop(i)
                    current_rects.insert(0, merged_rect)
                    merged_this = True
                    merged_any = True
                    break
            if not merged_this:
                next_rects.append(base)
        current_rects = next_rects

    return current_rects
