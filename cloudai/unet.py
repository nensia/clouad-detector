"""U-Net: нейросеть, которая смотрит на участки снимка целиком.

Обычная сеть (cloudai/net.py) решает по признакам каждого пикселя отдельно.
U-Net видит форму и соседство: облако — это целая область, а не набор точек.
Интерфейс тот же, что у CloudNet: analyze(), info, save(), load().
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from cloudai.features import compute_features, find_nodata
from cloudai.imageio import ImageReadError, load_image
from cloudai.net import CloudNet
from cloudai.synth import make_scene
from cloudai.training import LABEL_CLOUD, holdout_names, labeled_image_paths, read_label_map

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNET_PATH = ROOT / "models" / "cloud_unet.pt"
CROP = 128            # обучаем на случайных кусках 128×128: так быстрее, а смысл облака сохраняется
BATCH = 16
SYNTH_SIZE = 256


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Три уровня «сжать — расширить». Вход: снимок (3 канала), выход: оценка облачности по каждому пикселю."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.base = base
        self.enc1 = _block(3, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.bottom = _block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottom(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)[:, 0]                        # (N, H, W): «ответ» до сигмоиды


class CloudUNet:
    """Обёртка с тем же интерфейсом, что у CloudNet."""

    def __init__(self, module: UNet, info: dict | None = None):
        self.module = module.eval()
        self.info = info or {}

    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        """Вероятность облака для каждого пикселя снимка (высота × ширина)."""
        h, w = rgb.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8                       # U-Net работает с размерами, кратными 8
        padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else rgb
        x = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)).unsqueeze(0) - 0.5
        with torch.no_grad():
            prob = torch.sigmoid(self.module(x))[0].numpy()
        return prob[:h, :w].astype(np.float32)

    def analyze(self, rgb: np.ndarray, alpha: np.ndarray | None = None) -> dict:
        """Тот же результат, что у CloudNet.analyze: вероятность, маска облаков, данные, процент."""
        probability = ndimage.median_filter(self.predict_map(rgb), size=3, mode="nearest")
        valid = ~find_nodata(rgb, alpha)
        mask = (probability > 0.5) & valid
        n_valid = int(valid.sum())
        cloud_percent = 100.0 * float(mask.sum()) / n_valid if n_valid else 0.0
        return {"probability": probability, "mask": mask, "valid": valid,
                "cloud_percent": cloud_percent, "valid_fraction": n_valid / valid.size}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.module.state_dict(), "base": self.module.base}, path)
        path.with_suffix(".json").write_text(json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_UNET_PATH) -> "CloudUNet":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Нет файла модели {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        module = UNet(base=int(data["base"]))
        module.load_state_dict(data["state"])
        info_path = path.with_suffix(".json")
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        return cls(module, info)


# ------------------------------------------------------------------ проверка

def evaluate_maps(predict_map, items: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Совпадение с разметкой на снимках, которых модель не видела. Учитываются только пиксели с ответом."""
    tp = fp = fn = tn = 0
    for rgb, labels in items:
        known = labels > 0
        pred = predict_map(rgb)[known] > 0.5
        truth = labels[known] == LABEL_CLOUD
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
    total, union = tp + fp + fn + tn, tp + fp + fn
    return {"cloud_iou": round(tp / union, 4) if union else 0.0,
            "pixel_accuracy": round((tp + tn) / total, 4) if total else 0.0,
            "pixels": total}


def _reference(held_names: set[str], current_info_path: Path | None, base_path: Path | None, held_items):
    """С чем сравниваем новую модель на тех же отложенных снимках.

    Лучше всего — честный результат текущей модели (она при обучении эти снимки тоже не видела).
    Если его нет — базовая модель, обученная только на синтетике.
    """
    if current_info_path and Path(current_info_path).exists():
        info = json.loads(Path(current_info_path).read_text(encoding="utf-8"))
        ref = (info.get("holdout") or {}).get("candidate")
        if ref and set(info.get("holdout_images", [])) == held_names:
            return ref, "текущая модель"
    if base_path and Path(base_path).exists() and held_items:
        base = CloudNet.load(base_path)
        return evaluate_maps(lambda rgb: base.predict_proba(compute_features(rgb)), held_items), "базовая модель"
    return None, None


# ------------------------------------------------------------------ обучение

def _load_labeled(paths: list[Path], labels_dir: Path, masks_dir: Path | None):
    loaded = []
    for path in paths:
        try:
            image = load_image(path)
        except ImageReadError:
            continue
        labels = read_label_map(path, labels_dir, masks_dir, image["rgb"].shape[:2])
        if labels is not None:
            loaded.append((path.name, image["rgb"].astype(np.float32), labels))
    return loaded


def _as_training(loaded):
    """(rgb, ответ «облако», где есть ответ) для обучения. Снимки меньше куска пропускаем."""
    names, items, known_total = [], [], 0
    for name, rgb, labels in loaded:
        if min(rgb.shape[:2]) < CROP:
            continue
        known = labels > 0
        known_total += int(known.sum())
        names.append(name)
        items.append((rgb, (labels == LABEL_CLOUD).astype(np.float32), known.astype(np.float32)))
    return names, items, known_total


def _synthetic_pool(n: int, seed: int, log):
    pool = []
    for i in range(n):
        scene = make_scene(3_000_000 + seed * 100_000 + i, size=SYNTH_SIZE)   # номера не пересекаются с проверками
        pool.append((scene["rgb"].astype(np.float32), scene["mask"].astype(np.float32),
                     scene["valid"].astype(np.float32)))
        if log and (i + 1) % 60 == 0:
            log(f"     нарисовано учебных снимков: {i + 1} из {n}")
    return pool


def _random_crop(rgb, target, known, rng):
    h, w = target.shape
    y0 = int(rng.integers(0, h - CROP + 1))
    x0 = int(rng.integers(0, w - CROP + 1))
    r, t, k = (a[y0:y0 + CROP, x0:x0 + CROP] for a in (rgb, target, known))
    turns = int(rng.integers(4))
    r, t, k = np.rot90(r, turns), np.rot90(t, turns), np.rot90(k, turns)
    if rng.random() < 0.5:
        r, t, k = r[:, ::-1], t[:, ::-1], k[:, ::-1]
    r = np.clip(r * rng.uniform(0.85, 1.15), 0.0, 1.0)     # разная экспозиция снимков
    return (np.ascontiguousarray(r, dtype=np.float32), np.ascontiguousarray(t, dtype=np.float32),
            np.ascontiguousarray(k, dtype=np.float32))


def _batch(synth, real, rng):
    xs, ts, ks = [], [], []
    for i in range(BATCH):
        pool = synth if (not real or i % 2 == 0) else real           # половина кусков — ваши снимки
        rgb, target, known = pool[int(rng.integers(len(pool)))]
        r, t, k = _random_crop(rgb, target, known, rng)
        xs.append(r)
        ts.append(t)
        ks.append(k)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2) - 0.5
    return x.contiguous(), torch.from_numpy(np.stack(ts)), torch.from_numpy(np.stack(ks))


def _train(module: UNet, synth, real, steps: int, seed: int, log) -> UNet:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    module.train()
    started = time.time()
    for step in range(1, steps + 1):
        x, y, known = _batch(synth, real, rng)
        logits = module(x)
        loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss_map * known).sum() / known.sum().clamp(min=1.0)     # ошибка только там, где есть ответ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if log and (step % 200 == 0 or step == steps):
            log(f"     шаг {step} из {steps}, ошибка {loss.item():.3f} ({time.time() - started:.0f} с)")
    module.eval()
    return module


def train_unet_and_save(model_path: Path = DEFAULT_UNET_PATH, real_dir: Path | None = None,
                        labels_dir: Path | None = None, masks_dir: Path | None = None,
                        current_info_path: Path | None = None, base_path: Path | None = None,
                        steps: int = 800, n_synth: int = 240, seed: int = 0,
                        refit_all: bool = True, log=print) -> dict:
    """Учит U-Net, проверяет на отложенных снимках и сохраняет, только если она не хуже текущей модели."""
    real_dir = real_dir or ROOT / "data" / "real"
    labels_dir = labels_dir or ROOT / "data" / "labels"
    masks_dir = masks_dir or ROOT / "data" / "masks"
    started = time.time()

    labeled = labeled_image_paths(real_dir, labels_dir, masks_dir)
    held_names = holdout_names([p.name for p in labeled])
    loaded = _load_labeled(labeled, labels_dir, masks_dir)
    held_items = [(rgb, labels) for name, rgb, labels in loaded if name in held_names]
    train_names, real_train, real_pixels = _as_training([row for row in loaded if row[0] not in held_names])

    log(f"1/4. Готовлю снимки: учебных {n_synth}, ваших для обучения {len(train_names)}, "
        f"отложено для проверки {len(held_items)}")
    synth = _synthetic_pool(n_synth, seed, log)

    log(f"2/4. Обучаю U-Net, {steps} шагов…")
    candidate_model = CloudUNet(_train(UNet(), synth, real_train, steps, seed, log))

    log("3/4. Проверяю на снимках, которых сеть не видела…")
    candidate = evaluate_maps(candidate_model.predict_map, held_items) if held_items else None
    reference, source = _reference(held_names, current_info_path, base_path, held_items)
    if candidate and reference:
        log(f"     совпадение облаков: новая U-Net {candidate['cloud_iou'] * 100:.1f} %, "
            f"{source}: {reference['cloud_iou'] * 100:.1f} %")
    elif candidate:
        log(f"     совпадение облаков: {candidate['cloud_iou'] * 100:.1f} % (сравнивать не с чем)")
    promoted = candidate is None or reference is None or candidate["cloud_iou"] > reference["cloud_iou"]

    final, final_names, final_pixels = candidate_model, train_names, real_pixels
    if promoted and held_items and refit_all:
        # Проверка уже прошла: итоговую модель учим на всех размеченных снимках, включая отложенные.
        all_names, all_real, all_pixels = _as_training(loaded)
        log(f"     итоговая модель: дообучаю на всех снимках ({len(all_names)} шт.)…")
        final = CloudUNet(_train(UNet(), synth, all_real, steps, seed, log))
        final_names, final_pixels = all_names, all_pixels

    final.info = {
        "model": "U-Net",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "steps": steps,
        "synthetic_scenes": n_synth,
        "real_images": final_names,
        "real_pixels": int(final_pixels),
        "holdout_images": sorted(held_names),
        "holdout": {"candidate": candidate, "reference": reference, "reference_source": source},
        "promoted": promoted,
        "seconds": round(time.time() - started, 1),
    }
    log("4/4. Сохраняю модель…" if promoted else
        "4/4. Новая модель не лучше текущей на отдельных снимках — прежняя модель оставлена.")
    if promoted:
        final.save(model_path)
    log(f"Готово за {final.info['seconds']} с.")
    return final.info
