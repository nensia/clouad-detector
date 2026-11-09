"""Обучение нейросети.

Обучение — это подбор весов. Сети показывают пиксель (14 признаков) и правильный
ответ («облако» или «не облако»). Сеть ошибается, и алгоритм чуть-чуть двигает
веса так, чтобы в следующий раз ошибка стала меньше. Так сотни тысяч раз.

Откуда берутся примеры:
    1) учебные снимки из генератора (cloudai/synth.py) — там ответ известен точно;
    2) настоящие снимки, которые вы разметили во вкладке «Разметка» (data/labels);
    3) готовые маски, если они у вас есть (data/masks): белое — облако, чёрное — нет.

Честная проверка: часть ваших снимков сеть при обучении не видит. На них считается
совпадение облаков с вашей разметкой. Новая модель сохраняется, только если на этих
снимках она лучше базовой модели (той, что обучена лишь на синтетике).
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from cloudai.features import N_FEATURES, compute_features
from cloudai.imageio import ImageReadError, is_image_file, load_image
from cloudai.net import DEFAULT_MODEL_PATH, CloudNet
from cloudai.synth import make_scene

ROOT = Path(__file__).resolve().parent.parent
SYNTH_CACHE = ROOT / "models" / "synth_cache.npz"      # учебные пиксели для обучения «по умолчанию»

# Значения в файле разметки (data/labels/<имя снимка>.png)
LABEL_UNKNOWN, LABEL_CLEAR, LABEL_CLOUD = 0, 1, 2

# С одного снимка берём до стольких облачных и столько же чистых пикселей (поровну, а не вперемешку).
PER_CLASS_PER_IMAGE = 4000
# Каждый N-й снимок (по порядку дат) сеть при обучении не видит — на нём честно проверяем результат.
HOLDOUT_EVERY = 5


def build_synthetic_dataset(n_scenes: int = 600, pixels_per_scene: int = 400, seed: int = 0,
                            snow_share: float = 0.0, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Рисует n_scenes учебных снимков и берёт из каждого случайные пиксели с ответами."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    started = time.time()
    for i in range(n_scenes):
        scene = make_scene(seed * 1_000_003 + i, size=int(rng.choice([256, 320, 384])), snow_share=snow_share)
        feats = compute_features(scene["rgb"])[scene["valid"]]
        labels = scene["mask"][scene["valid"]]
        pick = rng.choice(len(labels), size=min(pixels_per_scene, len(labels)), replace=False)
        xs.append(feats[pick])
        ys.append(labels[pick])
        if log and (i + 1) % 100 == 0:
            log(f"  нарисовано снимков: {i + 1} из {n_scenes} ({time.time() - started:.0f} с)")
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.uint8)


def load_synthetic_cached(n_scenes: int = 600, seed: int = 0, snow_share: float = 0.0,
                          log=print) -> tuple[np.ndarray, np.ndarray]:
    """Учебные пиксели из сохранённого файла (чтобы дообучение не ждало генератор) или заново.

    Сохраняется только набор «по умолчанию»; для опытов с другими настройками снимки рисуются заново.
    """
    is_default = n_scenes == 600 and seed == 0 and snow_share == 0.0
    if is_default and SYNTH_CACHE.exists():
        data = np.load(SYNTH_CACHE, allow_pickle=False)
        if data["X"].shape[1] == N_FEATURES:
            return data["X"].astype(np.float32), data["y"]
    X, y = build_synthetic_dataset(n_scenes, seed=seed, snow_share=snow_share, log=log)
    if is_default:
        SYNTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SYNTH_CACHE, X=X.astype(np.float16), y=y)
    return X, y


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.name + ".png")


def labeled_image_paths(real_dir: Path, labels_dir: Path, masks_dir: Path | None = None) -> list[Path]:
    """Снимки из real_dir, для которых есть хоть какая-то разметка (мазки или полная маска)."""
    if not real_dir.exists():
        return []
    found = []
    for image_path in sorted(p for p in real_dir.iterdir() if is_image_file(p)):
        mask_file = (masks_dir / (image_path.stem + ".png")) if masks_dir else None
        if label_path_for(image_path, labels_dir).exists() or (mask_file and mask_file.exists()):
            found.append(image_path)
    return found


def read_label_map(image_path: Path, labels_dir: Path, masks_dir: Path | None, shape: tuple[int, int]):
    """Ответы на каждый пиксель снимка (0 — нет ответа, 1 — не облако, 2 — облако) или None, если разметки нет."""
    h, w = shape
    labels = np.zeros((h, w), dtype=np.uint8)
    mask_file = (masks_dir / (image_path.stem + ".png")) if masks_dir else None
    if mask_file and mask_file.exists():                      # полная маска: размечен каждый пиксель
        full = np.asarray(Image.open(mask_file).convert("L").resize((w, h), Image.NEAREST))
        labels = np.where(full > 127, LABEL_CLOUD, LABEL_CLEAR).astype(np.uint8)
    label_file = label_path_for(image_path, labels_dir)
    if label_file.exists():                                   # мазки из вкладки «Разметка» важнее маски
        scribble = np.asarray(Image.open(label_file).convert("L").resize((w, h), Image.NEAREST))
        labels = np.where(scribble > 0, scribble, labels).astype(np.uint8)
    if not np.any((labels == LABEL_CLEAR) | (labels == LABEL_CLOUD)):
        return None
    return labels


def image_pixels(image_path: Path, labels_dir: Path, masks_dir: Path | None = None):
    """Признаки всех пикселей снимка и его ответы (0 — нет ответа, 1 — не облако, 2 — облако).

    Возвращает None, если снимок не читается или разметки в нём нет.
    """
    try:
        image = load_image(image_path)
    except ImageReadError:
        return None
    labels = read_label_map(image_path, labels_dir, masks_dir, image["rgb"].shape[:2])
    if labels is None:
        return None
    return compute_features(image["rgb"]).reshape(-1, N_FEATURES), labels.reshape(-1)


def pick_balanced(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Случайные пиксели по классам: не больше per_class облачных и не больше per_class чистых."""
    picks = []
    for value in (LABEL_CLOUD, LABEL_CLEAR):
        idx = np.flatnonzero(labels == value)
        if len(idx):
            picks.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    return np.concatenate(picks) if picks else np.zeros(0, dtype=np.int64)


def load_real_samples(real_dir: Path, labels_dir: Path, masks_dir: Path | None = None,
                      max_per_image: int = 8000, seed: int = 1,
                      skip: set[str] | frozenset = frozenset()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Собирает размеченные пиксели настоящих снимков (поровну облачных и чистых).

    skip — имена снимков, которые не брать (например, отложенные для проверки).
    Возвращает (признаки, ответы, список использованных файлов).
    """
    rng = np.random.default_rng(seed)
    xs, ys, used = [], [], []
    for image_path in labeled_image_paths(real_dir, labels_dir, masks_dir):
        if image_path.name in skip:
            continue
        pixels = image_pixels(image_path, labels_dir, masks_dir)
        if pixels is None:
            continue
        feats, labels = pixels
        pick = pick_balanced(labels, max_per_image // 2, rng)
        xs.append(feats[pick])
        ys.append((labels[pick] == LABEL_CLOUD).astype(np.uint8))
        used.append(image_path.name)
    if not xs:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.uint8), used
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys), used


def holdout_names(names: list[str]) -> set[str]:
    """Снимки для честной проверки: каждый N-й по порядку имён (имена начинаются с даты съёмки)."""
    ordered = sorted(names)
    if len(ordered) < 4:
        return set()
    return set(ordered[::HOLDOUT_EVERY])


def evaluate_on_images(net, images: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Сравнивает ответ сети с разметкой на снимках, которых она не видела.

    Учитываются только пиксели с ответом. cloud_iou — совпадение облачных областей (1 = идеально).
    """
    tp = fp = fn = tn = 0
    for feats, labels in images:
        known = labels > 0
        pred = net.predict_proba(feats[known]) > 0.5
        truth = labels[known] == LABEL_CLOUD
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
    total, union = tp + fp + fn + tn, tp + fp + fn
    return {"cloud_iou": round(tp / union, 4) if union else 0.0,
            "pixel_accuracy": round((tp + tn) / total, 4) if total else 0.0,
            "pixels": total}


def fit_network(X: np.ndarray, y: np.ndarray, hidden: tuple[int, ...] = (32, 16), seed: int = 0,
                max_iter: int = 60) -> tuple[CloudNet, float]:
    """Подбирает веса сети. Возвращает сеть и точность на отложенных пикселях (0..1)."""
    from sklearn.neural_network import MLPClassifier       # scikit-learn нужен только для обучения

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xn))
    n_val = max(1000, len(Xn) // 8)                        # каждый восьмой пиксель сеть при обучении не видит
    val, train = order[:n_val], order[n_val:]
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4, batch_size=512,
                        learning_rate_init=2e-3, max_iter=max_iter, early_stopping=True,
                        n_iter_no_change=6, random_state=seed)
    clf.fit(Xn[train], y[train])
    accuracy = float(clf.score(Xn[val], y[val]))
    net = CloudNet(clf.coefs_, clf.intercepts_, mean, std)
    return net, accuracy


def _mix_real(X_syn: np.ndarray, y_syn: np.ndarray, Xr: np.ndarray, yr: np.ndarray):
    """Добавляет настоящие пиксели к учебным. Их повторяем, чтобы весили примерно как половина учебных."""
    if not len(yr):
        return X_syn, y_syn, 0
    repeat = int(np.clip(round(0.5 * len(y_syn) / len(yr)), 1, 50))
    return np.concatenate([X_syn] + [Xr] * repeat), np.concatenate([y_syn] + [yr] * repeat), repeat


def train_and_save(model_path: Path = DEFAULT_MODEL_PATH, n_scenes: int = 600, seed: int = 0,
                   snow_share: float = 0.0, real_dir: Path | None = None, labels_dir: Path | None = None,
                   masks_dir: Path | None = None, log=print) -> dict:
    """Учит сеть, проверяет на отдельных снимках и сохраняет, только если она лучше базовой."""
    real_dir = real_dir or ROOT / "data" / "real"
    labels_dir = labels_dir or ROOT / "data" / "labels"
    masks_dir = masks_dir or ROOT / "data" / "masks"
    started = time.time()

    log("1/4. Готовлю учебные снимки из генератора…")
    X_syn, y_syn = load_synthetic_cached(n_scenes, seed, snow_share, log)

    labeled = labeled_image_paths(real_dir, labels_dir, masks_dir)
    held_names = holdout_names([p.name for p in labeled])
    Xr, yr, used = load_real_samples(real_dir, labels_dir, masks_dir, skip=held_names)
    X, y, repeat = _mix_real(X_syn, y_syn, Xr, yr)
    if held_names:
        log(f"     отложено для проверки: {len(held_names)} снимков (сеть их не учит)")
    if len(yr):
        log(f"     + настоящие снимки: {len(used)} шт., размеченных пикселей: {len(yr)} (повторены ×{repeat})")
    else:
        log("     настоящих размеченных снимков пока нет — учусь только на учебных")

    log(f"2/4. Обучаю нейросеть на {len(y)} пикселях…")
    net, accuracy = fit_network(X, y, seed=seed)

    held = []
    for path in labeled:
        if path.name in held_names:
            pixels = image_pixels(path, labels_dir, masks_dir)
            if pixels is not None:
                held.append(pixels)
    candidate = evaluate_on_images(net, held) if held else None
    base_path = model_path.with_name("cloud_net_base.npz")
    reference = None
    reference_source = None
    current_info_path = model_path.with_suffix(".json")
    if current_info_path.exists():
        try:
            current_info = json.loads(current_info_path.read_text(encoding="utf-8"))
            prior_holdout = (current_info.get("holdout") or {}).get("candidate")
            if prior_holdout and set(current_info.get("holdout_images", [])) == held_names:
                reference = prior_holdout
                reference_source = "текущая модель"
        except (OSError, ValueError, TypeError):
            pass
    if reference is None and held and base_path.exists():
        reference = evaluate_on_images(CloudNet.load(base_path), held)
        reference_source = "базовая модель"
    promoted = candidate is None or reference is None or candidate["cloud_iou"] > reference["cloud_iou"]
    log("3/4. Проверяю на снимках, которых сеть не видела…")
    if candidate and reference:
        log(f"     совпадение облаков: новая модель {candidate['cloud_iou'] * 100:.1f} %, "
            f"{reference_source} {reference['cloud_iou'] * 100:.1f} %")
    elif candidate:
        log(f"     совпадение облаков: {candidate['cloud_iou'] * 100:.1f} % (базовой модели для сравнения нет)")

    final_Xr, final_yr, final_used = Xr, yr, used
    if promoted and held_names:
        # Снимки проверки уже сыграли свою роль: итоговую модель учим на всех размеченных снимках.
        final_Xr, final_yr, final_used = load_real_samples(real_dir, labels_dir, masks_dir)
        X, y, repeat = _mix_real(X_syn, y_syn, final_Xr, final_yr)
        log(f"     итоговая модель: дообучаю на всех снимках ({len(final_used)} шт.), пикселей {len(y)}…")
        net, accuracy = fit_network(X, y, seed=seed)

    net.info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "synthetic_scenes": n_scenes,
        "snow_share": snow_share,
        "synthetic_pixels": int(len(y_syn)),
        "real_pixels": int(len(final_yr)),
        "real_images": final_used,
        "holdout_images": sorted(held_names),
        "holdout": {"candidate": candidate, "reference": reference,
                    "reference_source": reference_source},
        "promoted": promoted,
        "pixel_accuracy_holdout": round(accuracy, 4),
        "seconds": round(time.time() - started, 1),
    }
    log("4/4. Сохраняю модель…" if promoted else
        "4/4. Новая модель не лучше базовой на отдельных снимках — прежняя модель оставлена.")
    if promoted:
        net.save(model_path)
    log(f"Готово за {net.info['seconds']} с.")
    return net.info
