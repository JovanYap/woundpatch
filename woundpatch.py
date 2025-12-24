#!/usr/bin/env python3
"""Detect a wound-like region and overlay a bandage."""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.mixture import GaussianMixture

try:
    import torch
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None
    torch = None


def read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def largest_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    hsv_mask1 = cv2.inRange(hsv, (0, 40, 60), (25, 255, 255))
    hsv_mask2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
    hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)

    ycrcb_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return largest_contour_mask(mask)


def estimate_arm_angle(mask: np.ndarray) -> float:
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] < 50:
        return 0.0

    points = points.astype(np.float32)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, order[0]]

    angle = math.atan2(principal[0], principal[1])
    return angle


def detect_wound_heuristic(bgr: np.ndarray, skin: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    a_channel = lab[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)

    skin_pixels = skin > 0
    if not np.any(skin_pixels):
        h, w = bgr.shape[:2]
        return [((w // 2, h // 2), max(60, w // 6), max(20, w // 18))]

    a_vals = a_channel[skin_pixels]
    v_vals = v_channel[skin_pixels]
    a_mean, a_std = float(np.mean(a_vals)), float(np.std(a_vals) + 1e-6)
    v_mean, v_std = float(np.mean(v_vals)), float(np.std(v_vals) + 1e-6)

    red_score = (a_channel - a_mean) / a_std
    dark_score = (v_channel - v_mean) / v_std

    wound_mask = (red_score > 1.0) | (dark_score < -0.6)
    wound_mask = wound_mask.astype(np.uint8) * 255
    wound_mask = cv2.bitwise_and(wound_mask, skin)

    kernel = np.ones((5, 5), np.uint8)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(skin_pixels)
        center = (int(xs.mean()), int(ys.mean()))
        return [(center, 100, 34)]

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    center = (x + w // 2, y + h // 2)

    bandage_w = max(60, int(2.6 * max(w, h)))
    bandage_h = max(20, int(bandage_w / 3))

    return [(center, bandage_w, bandage_h)]


def detect_wound_gmm(bgr: np.ndarray, skin: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    texture = cv2.Laplacian(gray, cv2.CV_32F)
    texture = cv2.GaussianBlur(np.abs(texture), (3, 3), 0)

    skin_pixels = skin > 0
    if not np.any(skin_pixels):
        h, w = bgr.shape[:2]
        return [((w // 2, h // 2), max(60, w // 6), max(20, w // 18))]

    a_vals = a_channel[skin_pixels]
    b_vals = b_channel[skin_pixels]
    v_vals = v_channel[skin_pixels]
    t_vals = texture[skin_pixels]

    features = np.stack([a_vals, b_vals, v_vals, t_vals], axis=1)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features_norm = (features - mean) / std

    sample_size = min(5000, features_norm.shape[0])
    if sample_size < 200:
        return []

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(features_norm.shape[0], size=sample_size, replace=False)
    sample = features_norm[sample_idx]

    n_components = 4 if sample_size >= 1500 else 3
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(sample)

    labels = gmm.predict(features_norm)
    probs = gmm.predict_proba(features_norm)
    component_scores = []
    component_stats: list[tuple[float, float, float, float]] = []
    for idx in range(n_components):
        comp = features_norm[labels == idx]
        if comp.size == 0:
            component_scores.append(-np.inf)
            component_stats.append((0.0, 0.0, 0.0, 0.0))
            continue
        z_a = comp[:, 0].mean()
        z_v = comp[:, 2].mean()
        z_t = comp[:, 3].mean()
        comp_mean_prob = float(probs[labels == idx, idx].mean())
        score = 0.6 * z_a + 0.2 * z_t - 0.2 * z_v
        component_scores.append(score)
        component_stats.append((z_a, z_v, z_t, comp_mean_prob))

    scored_components = np.argsort(component_scores)[::-1]
    selected = []
    skin_count = float(np.sum(skin_pixels))
    for idx in scored_components[:1]:
        comp_count = float(np.sum(labels == idx))
        if comp_count == 0:
            continue
        z_a, z_v, z_t, comp_mean_prob = component_stats[idx]
        comp_ratio = comp_count / skin_count
        score = component_scores[idx]
        if (
            comp_mean_prob >= 0.75
            and score >= 0.85
            and comp_ratio >= 0.001
            and ((z_a > 1.0) or (z_v < -0.8) or (z_t > 1.0))
        ):
            selected.append(idx)

    if not selected:
        return []

    wound_mask = np.zeros_like(skin, dtype=np.uint8)
    all_idx = np.where(skin_pixels)
    prob_threshold = 0.75
    for comp_id in selected:
        comp_pixels = (labels == comp_id) & (probs[:, comp_id] >= prob_threshold)
        wound_mask[all_idx[0][comp_pixels], all_idx[1][comp_pixels]] = 255

    kernel = np.ones((5, 5), np.uint8)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    skin_area = float(np.sum(skin_pixels))
    min_area = max(80.0, skin_area * 0.001)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        bandage_w = max(50, int(2.4 * max(w, h)))
        bandage_h = max(20, int(bandage_w / 3))
        results.append((center, bandage_w, bandage_h))

    return results


_FASTSAM_MODEL = None
_WOUNDSEG_MODEL = None


def get_fastsam_model() -> "YOLO":
    global _FASTSAM_MODEL
    if _FASTSAM_MODEL is None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed; install requirements.txt")
        local_path = os.path.join(os.path.dirname(__file__), "fastsam-s.pt")
        model_path = local_path if os.path.isfile(local_path) else "fastsam-s.pt"
        _FASTSAM_MODEL = YOLO(model_path)
    return _FASTSAM_MODEL


def get_woundseg_model():
    global _WOUNDSEG_MODEL
    if _WOUNDSEG_MODEL is None:
        base_dir = os.path.join(os.path.dirname(__file__), "wound-segmentation")
        weights_path = os.path.join(
            base_dir, "training_history", "2019-12-19 01%3A53%3A15.480800.hdf5"
        )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing wound-segmentation weights at {weights_path}"
            )
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        try:
            import keras
            from keras.models import load_model

            # Patch missing conv_utils for keras>=2.15 used by wound-segmentation repo.
            if not hasattr(keras.utils, "conv_utils"):
                import types

                def normalize_tuple(value, n, name):
                    if isinstance(value, int):
                        return (value,) * n
                    if isinstance(value, (tuple, list)) and len(value) == n:
                        try:
                            return tuple(int(v) for v in value)
                        except Exception as exc:
                            raise ValueError(f"{name} must be a tuple of {n} ints") from exc
                    raise ValueError(f"{name} must be a tuple of {n} ints")

                conv_utils = types.SimpleNamespace(normalize_tuple=normalize_tuple)
                keras.utils.conv_utils = conv_utils
                sys.modules["keras.utils.conv_utils"] = conv_utils

            if not hasattr(keras.utils, "data_utils"):
                import types

                data_utils = types.SimpleNamespace(get_file=keras.utils.get_file)
                keras.utils.data_utils = data_utils
                sys.modules["keras.utils.data_utils"] = data_utils

            from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
            from utils.learning.metrics import dice_coef, precision, recall
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TensorFlow/Keras is required for woundseg mode (Keras import failed)"
            ) from exc

        model = Deeplabv3(weights=None, input_shape=(224, 224, 3), classes=1)
        model = load_model(
            weights_path,
            custom_objects={
                "recall": recall,
                "precision": precision,
                "dice_coef": dice_coef,
                "relu6": relu6,
                "DepthwiseConv2D": DepthwiseConv2D,
                "BilinearUpsampling": BilinearUpsampling,
            },
        )
        _WOUNDSEG_MODEL = model
    return _WOUNDSEG_MODEL


def score_masks(
    masks: list[np.ndarray],
    a_channel: np.ndarray,
    v_channel: np.ndarray,
    texture: np.ndarray,
    skin: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    skin_pixels = skin > 0
    if not np.any(skin_pixels):
        return []

    a_vals = a_channel[skin_pixels]
    v_vals = v_channel[skin_pixels]
    t_vals = texture[skin_pixels]
    mean = np.array([a_vals.mean(), v_vals.mean(), t_vals.mean()])
    std = np.array([a_vals.std(), v_vals.std(), t_vals.std()]) + 1e-6

    scored = []
    for mask in masks:
        mask = (mask > 0) & skin_pixels
        if mask.sum() < 80:
            continue
        a_m = a_channel[mask].mean()
        v_m = v_channel[mask].mean()
        t_m = texture[mask].mean()
        z = (np.array([a_m, v_m, t_m]) - mean) / std
        score = 0.6 * z[0] + 0.25 * z[2] - 0.15 * z[1]
        scored.append((mask.astype(np.uint8) * 255, float(score)))

    return scored


def masks_to_bandages(mask: np.ndarray, skin: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    if mask.sum() == 0:
        return []
    kernel = np.ones((9, 9), np.uint8)
    merged = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    skin_area = float(np.sum(skin > 0))
    min_area = max(80.0, skin_area * 0.001)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        bandage_w = max(50, int(2.4 * max(w, h)))
        bandage_h = max(20, int(bandage_w / 3))
        results.append((center, bandage_w, bandage_h))
    return results


def bandage_from_mask(mask: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return []
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min
    h = y_max - y_min
    center = (x_min + w // 2, y_min + h // 2)
    bandage_w = max(60, int(2.4 * max(w, h)))
    bandage_h = max(20, int(bandage_w / 3))
    return [(center, bandage_w, bandage_h)]


def detect_wound_fastsam(bgr: np.ndarray, skin: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    if YOLO is None or torch is None:
        raise ImportError("ultralytics/torch not available; install requirements.txt")

    model = get_fastsam_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source=rgb,
        imgsz=1024,
        conf=0.25,
        iou=0.9,
        device=device,
        verbose=False,
    )
    if not results or results[0].masks is None:
        return []

    mask_tensor = results[0].masks.data
    masks = []
    for idx in range(mask_tensor.shape[0]):
        mask = mask_tensor[idx].cpu().numpy()
        if mask.shape[:2] != skin.shape[:2]:
            mask = cv2.resize(mask, (skin.shape[1], skin.shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 0.5).astype(np.uint8))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    a_channel = lab[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    texture = cv2.Laplacian(gray, cv2.CV_32F)
    texture = cv2.GaussianBlur(np.abs(texture), (3, 3), 0)

    scored = score_masks(masks, a_channel, v_channel, texture, skin)
    if not scored:
        return []

    scored.sort(key=lambda item: item[1], reverse=True)
    selected_masks = [m for m, s in scored if s >= 0.9]
    if not selected_masks:
        selected_masks = [scored[0][0]]

    combined = np.zeros_like(skin, dtype=np.uint8)
    for mask in selected_masks:
        combined = cv2.bitwise_or(combined, mask)

    return masks_to_bandages(combined, skin)


def detect_wound_woundseg(
    bgr: np.ndarray, skin: np.ndarray
) -> tuple[list[tuple[tuple[int, int], int, int]], np.ndarray]:
    model = get_woundseg_model()
    input_size = (224, 224)
    resized = cv2.resize(bgr, input_size, interpolation=cv2.INTER_LINEAR).astype("float32")
    diff = float(resized.max() - resized.min())
    resized = resized / (diff if diff > 0 else 255.0)
    pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0, :, :, 0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.bitwise_and(mask, skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if mask.sum() == 0:
        return [], np.zeros_like(skin, dtype=np.uint8)

    connect_kernel = np.ones((15, 15), np.uint8)
    connected = cv2.dilate(mask, connect_kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1:
        return bandage_from_mask(connected), mask

    return masks_to_bandages(mask, skin), mask


def make_segmentation_overlay(bgr: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = [220, 60, 60]
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    return Image.fromarray(blended)


def make_bandage(width: int, height: int, variant: int = 0) -> Image.Image:
    width = max(10, width)
    height = max(10, height)

    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    radius = int(min(width, height) * 0.25)
    palette = [
        ((230, 206, 170, 240), (245, 233, 210, 255)),
        ((220, 198, 160, 240), (240, 228, 205, 255)),
        ((235, 212, 176, 240), (248, 236, 214, 255)),
    ]
    adhesive, pad = palette[variant % len(palette)]

    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        fill=adhesive,
        outline=(210, 186, 150, 255),
        width=2,
    )

    pad_margin_x = int(width * 0.28)
    pad_margin_y = int(height * 0.2)
    draw.rounded_rectangle(
        [(pad_margin_x, pad_margin_y), (width - pad_margin_x, height - pad_margin_y)],
        radius=int(radius * 0.6),
        fill=pad,
    )

    return base


def overlay_bandage(
    bgr: np.ndarray,
    placements: list[tuple[tuple[int, int], int, int]],
    angle_rad: float,
    bandage_path: str | None,
) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb)

    for idx, (center, bandage_w, bandage_h) in enumerate(placements):
        if bandage_path and os.path.isfile(bandage_path):
            bandage = Image.open(bandage_path).convert("RGBA")
            bandage = bandage.resize((bandage_w, bandage_h), Image.LANCZOS)
        else:
            bandage = make_bandage(bandage_w, bandage_h, variant=idx)

        angle_deg = math.degrees(angle_rad)
        rotated = bandage.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

        x = int(center[0] - rotated.size[0] / 2)
        y = int(center[1] - rotated.size[1] / 2)

        base.paste(rotated, (x, y), rotated)
    return base


def save_side_by_side(original: Image.Image, patched: Image.Image, path: str) -> None:
    width = original.width + patched.width
    height = max(original.height, patched.height)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    canvas.paste(original, (0, 0))
    canvas.paste(patched, (original.width, 0))
    canvas.save(path)


def process_image(
    path: str,
    output_dir: str,
    bandage_path: str | None,
    show: bool,
    wound_mode: str,
) -> None:
    bgr = read_bgr(path)
    skin = skin_mask(bgr)
    angle = estimate_arm_angle(skin)
    if wound_mode == "gmm":
        placements = detect_wound_gmm(bgr, skin)
    elif wound_mode == "fastsam":
        placements = detect_wound_fastsam(bgr, skin)
    elif wound_mode == "woundseg":
        placements, seg_mask = detect_wound_woundseg(bgr, skin)
    else:
        placements = detect_wound_heuristic(bgr, skin)

    if not placements:
        print(f"No high-confidence wound detected for {path}")

    patched = overlay_bandage(bgr, placements, angle, bandage_path)
    original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{stem}_bandaged_{wound_mode}.png")
    compare_path = os.path.join(output_dir, f"{stem}_compare_{wound_mode}.png")

    patched.save(out_path)
    save_side_by_side(original, patched, compare_path)

    print(f"Saved: {out_path}")
    print(f"Saved: {compare_path}")

    if show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(patched)
        axes[1].set_title("Bandaged")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a wound and overlay a bandage.")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("--output-dir", default="outputs", help="Where to save results")
    parser.add_argument("--bandage", default=None, help="Optional bandage PNG (with alpha)")
    parser.add_argument(
        "--wound-mode",
        choices=["heuristic", "gmm", "fastsam", "woundseg"],
        default="gmm",
        help="Wound detection mode (default: gmm).",
    )
    parser.add_argument("--show", action="store_true", help="Show results with matplotlib")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in args.images:
        try:
            process_image(path, args.output_dir, args.bandage, args.show, args.wound_mode)
        except Exception as exc:
            print(f"Error processing {path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
