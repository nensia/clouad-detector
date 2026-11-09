"""Общие заготовки для тестов.  Запуск всех тестов:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloudai.net import CloudNet  # noqa: E402
from cloudai.synth import make_scene, save_scene  # noqa: E402


@pytest.fixture(scope="session")
def model() -> CloudNet:
    return CloudNet.load()


@pytest.fixture(scope="session")
def active_model():
    """Модель, которой реально пользуется программа: принятая U-Net, а если её нет — обычная сеть."""
    from cloudai.net import load_active_model
    return load_active_model()


@pytest.fixture
def scene_file(tmp_path):
    """Фабрика: создаёт учебный снимок с заданной долей облаков и возвращает (путь, истинный процент)."""
    def make(name: str, coverage: float, seed: int = 1, folder: Path | None = None, kind: str = "лес и поля"):
        folder = folder or tmp_path
        folder.mkdir(parents=True, exist_ok=True)
        scene = make_scene(seed, size=256, coverage=coverage, kind=kind, nodata=False, cloud_style="слоистые")
        path = folder / name
        save_scene(scene, str(path))
        return path, scene["percent"]
    return make


def flat_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (200, 160)) -> Path:
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(path)
    return path
