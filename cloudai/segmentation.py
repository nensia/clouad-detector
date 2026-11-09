"""Edge-aware flood fill helpers for cloud annotation."""
from collections import deque

import numpy as np
from scipy import ndimage


def edge_aware_component(candidate: np.ndarray, rgb: np.ndarray, seed_x: int, seed_y: int,
                         tolerance: float = 0.24) -> np.ndarray:
    """Return the candidate region reachable from a seed without crossing strong color edges.

    ``candidate`` is the semantic class predicted by the model. Flooding only within
    that mask preserves the model's cloud-vs-clear decision; per-edge RGB and
    brightness checks then prevent the fill from crossing visible object boundaries.
    Tolerance is a normalized channel delta in [0, 1].
    """
    candidate = np.asarray(candidate, dtype=bool)
    rgb = np.asarray(rgb, dtype=np.float32)
    if candidate.ndim != 2 or rgb.shape != candidate.shape + (3,):
        raise ValueError("candidate must be H×W and rgb must be H×W×3")
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be between 0 and 1")

    height, width = candidate.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height) or not candidate[seed_y, seed_x]:
        return np.zeros(candidate.shape, dtype=bool)

    # Limit traversal to the model's connected semantic region first.
    components, _ = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    component_id = components[seed_y, seed_x]
    component = (components == component_id).ravel()
    rgb_flat = rgb.reshape(-1, 3)
    brightness = rgb.mean(axis=-1).ravel()

    # Keep brightness changes a little stricter than color changes: cloud edges
    # often show up as a luminance jump even when the hue is nearly unchanged.
    brightness_tolerance = tolerance * 1.25
    visited = np.zeros(height * width, dtype=bool)
    start = seed_y * width + seed_x
    visited[start] = True
    queue: deque[int] = deque([start])

    while queue:
        current = queue.popleft()
        x, y = current % width, current // width
        neighbors = []
        if x > 0:
            neighbors.append(current - 1)
        if x + 1 < width:
            neighbors.append(current + 1)
        if y > 0:
            neighbors.append(current - width)
        if y + 1 < height:
            neighbors.append(current + width)

        current_rgb = rgb_flat[current]
        current_brightness = brightness[current]
        for neighbor in neighbors:
            if visited[neighbor] or not component[neighbor]:
                continue
            color_delta = float(np.max(np.abs(rgb_flat[neighbor] - current_rgb)))
            brightness_delta = abs(float(brightness[neighbor] - current_brightness))
            if color_delta <= tolerance and brightness_delta <= brightness_tolerance:
                visited[neighbor] = True
                queue.append(neighbor)

    return visited.reshape(height, width)
