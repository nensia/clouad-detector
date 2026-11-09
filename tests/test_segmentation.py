import numpy as np

from cloudai.segmentation import edge_aware_component


def test_edge_aware_component_stops_at_strong_color_and_brightness_edge():
    candidate = np.ones((8, 10), dtype=bool)
    rgb = np.zeros((8, 10, 3), dtype=np.float32)
    rgb[:, :5] = 0.82
    rgb[:, 5:] = 0.12

    region = edge_aware_component(candidate, rgb, seed_x=2, seed_y=4, tolerance=0.24)

    assert region[:, :5].all()
    assert not region[:, 5:].any()


def test_edge_aware_component_keeps_smooth_color_transition_connected():
    candidate = np.ones((3, 7), dtype=bool)
    levels = np.linspace(0.2, 0.8, candidate.shape[1], dtype=np.float32)
    rgb = np.repeat(levels[None, :, None], candidate.shape[0], axis=0)
    rgb = np.repeat(rgb, 3, axis=2)

    region = edge_aware_component(candidate, rgb, seed_x=0, seed_y=1, tolerance=0.12)

    assert region.all()


def test_edge_aware_component_returns_empty_for_invalid_seed():
    candidate = np.ones((4, 4), dtype=bool)
    candidate[1, 1] = False
    region = edge_aware_component(candidate, np.ones((4, 4, 3), dtype=np.float32), 1, 1)

    assert not region.any()
