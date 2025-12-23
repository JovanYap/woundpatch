#!/usr/bin/env python3
"""Detect a wound-like region and overlay a bandage."""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw


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


def detect_wound(bgr: np.ndarray, skin: np.ndarray) -> tuple[tuple[int, int], int, int]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    a_channel = lab[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)

    skin_pixels = skin > 0
    if not np.any(skin_pixels):
        h, w = bgr.shape[:2]
        return (w // 2, h // 2), max(60, w // 6), max(20, w // 18)

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
        return center, 100, 34

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    center = (x + w // 2, y + h // 2)

    bandage_w = max(60, int(2.6 * max(w, h)))
    bandage_h = max(20, int(bandage_w / 3))

    return center, bandage_w, bandage_h


def make_bandage(width: int, height: int) -> Image.Image:
    width = max(10, width)
    height = max(10, height)

    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    radius = int(min(width, height) * 0.25)
    adhesive = (230, 206, 170, 240)
    pad = (245, 233, 210, 255)

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
    center: tuple[int, int],
    bandage_size: tuple[int, int],
    angle_rad: float,
    bandage_path: str | None,
) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb)

    if bandage_path and os.path.isfile(bandage_path):
        bandage = Image.open(bandage_path).convert("RGBA")
        bandage = bandage.resize(bandage_size, Image.LANCZOS)
    else:
        bandage = make_bandage(*bandage_size)

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


def process_image(path: str, output_dir: str, bandage_path: str | None, show: bool) -> None:
    bgr = read_bgr(path)
    skin = skin_mask(bgr)
    angle = estimate_arm_angle(skin)
    center, bandage_w, bandage_h = detect_wound(bgr, skin)

    patched = overlay_bandage(bgr, center, (bandage_w, bandage_h), angle, bandage_path)
    original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{stem}_bandaged.png")
    compare_path = os.path.join(output_dir, f"{stem}_compare.png")

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
    parser.add_argument("--show", action="store_true", help="Show results with matplotlib")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in args.images:
        try:
            process_image(path, args.output_dir, args.bandage, args.show)
        except Exception as exc:
            print(f"Error processing {path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
