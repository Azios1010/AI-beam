"""Bounding-box projection and rectangle geometry tests."""
import pytest

from aura_beam import (
    ControlTarget, GridRectangle, MatrixGridController,
    merge_rectangles, rectangles_to_mask,
)


def test_rectangles_to_mask_supports_custom_grid() -> None:
    mask = rectangles_to_mask([GridRectangle(10, 12, 3, 4)], width=16, height=16)
    assert mask.shape == (16, 16)
    assert int(mask.sum()) == 6


def test_merge_rectangles_merges_adjacent_regions() -> None:
    assert merge_rectangles([GridRectangle(1, 2, 1, 2), GridRectangle(3, 4, 1, 2)]) == [GridRectangle(1, 4, 1, 2)]


def test_rectangles_to_mask_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValueError, match="outside 8x8 grid"):
        rectangles_to_mask([GridRectangle(7, 8, 1, 2)], width=8, height=8)


def test_controller_projects_exact_cell_boundaries_without_extra_cells() -> None:
    regions = MatrixGridController().get_dark_rectangles(
        [ControlTarget(25, 15, 10, 10)], screen_width=80, screen_height=80,
    )
    assert regions == [GridRectangle(2, 2, 1, 1)]


def test_controller_clips_box_to_image_boundaries() -> None:
    regions = MatrixGridController().get_dark_rectangles(
        [ControlTarget(0, 0, 20, 20)], screen_width=80, screen_height=80,
    )
    assert regions == [GridRectangle(0, 0, 0, 0)]


def test_controller_maps_whole_image_to_whole_grid() -> None:
    regions = MatrixGridController(16, 16).get_dark_rectangles(
        [ControlTarget(40, 40, 80, 80)], screen_width=80, screen_height=80,
    )
    assert regions == [GridRectangle(0, 15, 0, 15)]


def test_controller_ignores_boxes_fully_outside_image() -> None:
    regions = MatrixGridController().get_dark_rectangles(
        [ControlTarget(-20, 40, 10, 10)], screen_width=80, screen_height=80,
    )
    assert regions == []


@pytest.mark.parametrize("target", [ControlTarget(10, 10, 0, 10), ControlTarget(float("nan"), 10, 10, 10)])
def test_controller_rejects_invalid_boxes(target: ControlTarget) -> None:
    with pytest.raises(ValueError):
        MatrixGridController().get_dark_rectangles([target], screen_width=80, screen_height=80)
