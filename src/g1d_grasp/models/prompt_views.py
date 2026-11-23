from dataclasses import dataclass

import cv2
import numpy as np


PROMPT_AGE_CAP_S = 0.2


@dataclass(frozen=True)
class Crop:
    center_xy: tuple[float, float]
    side_px: float


@dataclass(frozen=True)
class View:
    channels: np.ndarray
    raw_to_view: np.ndarray
    view_to_raw: np.ndarray
    valid_pixels: np.ndarray
    point_visible: bool


def encode_views(rgb, bbox, point, *, min_crop_px, previous=None, size=224, sigma=4):
    """Use only observed RGB and normalized caller/tracker prompts."""
    rgb = np.asarray(rgb)
    bbox = np.asarray(bbox, dtype=float)
    point = np.asarray(point, dtype=float)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 1:
        raise ValueError("Expected raw HWC uint8 RGB")
    if bbox.shape != (4,) or point.shape != (2,) or not np.isfinite(np.r_[bbox, point]).all():
        raise ValueError("Invalid bounding box or point")
    if np.any(np.r_[bbox, point] < 0) or np.any(np.r_[bbox, point] > 1) or not np.all(bbox[:2] < bbox[2:]):
        raise ValueError("Box and point must lie on the normalized original image")
    if size < 2 or not np.isfinite([min_crop_px, sigma]).all() or min_crop_px <= 0 or sigma <= 0:
        raise ValueError("Invalid crop or heatmap parameters")
    height, width = rgb.shape[:2]
    limits = np.array([width - 1, height - 1], dtype=float)
    box_pixels = bbox.reshape(2, 2) * limits
    point_pixels = point * limits
    raw_side = max(float(np.max(box_pixels[1] - box_pixels[0])) * 1.5, min_crop_px)
    center = box_pixels.mean(axis=0)
    if previous:
        center = 0.7 * center + 0.3 * np.asarray(previous.center_xy)
        raw_side = 0.7 * raw_side + 0.3 * previous.side_px
    if not np.isfinite(np.r_[center, raw_side]).all() or raw_side <= 0:
        raise ValueError("Invalid temporal crop state")
    side = max(raw_side, 2 * float(np.max(np.abs(point_pixels - center))) + 2)
    crop = Crop(tuple(center.tolist()), side)
    global_scale = min((size - 1) / limits)
    global_view = render(rgb, box_pixels, point_pixels, limits / 2, global_scale, size, sigma)
    local_view = render(rgb, box_pixels, point_pixels, center, (size - 1) / side, size, sigma)
    return global_view, local_view, crop


def render(rgb, bbox, point, center, scale, size, sigma):
    offset = (size - 1) / 2 - np.asarray(center) * scale
    affine = np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]])
    inverse = np.linalg.inv(affine)
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    source_x = (x_grid - offset[0]) / scale
    source_y = (y_grid - offset[1]) / scale
    height, width = rgb.shape[:2]
    valid = (source_x >= -1e-8) & (source_x <= width - 1 + 1e-8)
    valid &= (source_y >= -1e-8) & (source_y <= height - 1 + 1e-8)
    pixels = cv2.warpAffine(rgb, affine[:2], (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    pixels[~valid] = 0
    box = bbox * scale + offset
    mask = ((x_grid >= box[0, 0]) & (x_grid <= box[1, 0]) & (y_grid >= box[0, 1]) & (y_grid <= box[1, 1]) & valid)
    point_x, point_y = point * scale + offset
    visible = bool(0 <= point_x <= size - 1 and 0 <= point_y <= size - 1)
    heatmap = np.exp(-((x_grid - point_x) ** 2 + (y_grid - point_y) ** 2) / (2 * sigma**2)) * valid if visible else np.zeros_like(x_grid)
    channels = np.concatenate((pixels.transpose(2, 0, 1).astype(np.float32) / 255,
                               mask[None].astype(np.float32), heatmap[None].astype(np.float32)))
    return View(channels, affine, inverse, valid, visible)
