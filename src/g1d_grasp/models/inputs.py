import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .prompt_views import render


def rgb224(rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB")
    return rgb if rgb.shape[:2] == (224, 224) else cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


def proprioception(qpos, width, eef):
    qpos, width, eef = np.asarray(qpos), np.asarray(width), np.asarray(eef)
    if qpos.shape[-1:] != (7,) or width.shape != qpos.shape[:-1] or eef.shape != (*width.shape, 7):
        raise ValueError("Expected aligned arm, jaw aperture and xyz/xyzw pose")
    rotation = Rotation.from_quat(eef[..., 3:].reshape(-1,4)).as_matrix()
    columns = rotation[:, :, :2].swapaxes(-1, -2).reshape(*width.shape, 6)
    return np.concatenate((qpos, width[..., None], eef[..., :3], columns), axis=-1).astype(np.float32)


def head_input(rgb, bbox, point, size=224, *, point_geometry=False):
    height, width = rgb.shape[:2]
    limits = np.array([width - 1, height - 1])
    bbox = np.asarray(bbox, dtype=float)
    point = np.asarray(point, dtype=float)
    if (bbox.shape != (4,) or point.shape != (2,) or not np.isfinite(np.r_[bbox, point]).all()
            or np.any(np.r_[bbox, point] < 0) or np.any(np.r_[bbox, point] > 1) or np.any(bbox[:2] >= bbox[2:])):
        raise ValueError("Invalid first-frame head prompt")
    bounds = bbox.reshape(2, 2) * limits
    scale = min((size - 1) / limits)
    view = render(rgb, bounds, point * limits, limits / 2, scale, size, 4)
    center = bbox.reshape(2, 2).mean(axis=0)
    side = bbox[2:] - bbox[:2]
    meta = [*center, *side, 1, 0]
    if point_geometry:
        meta.extend([*point, *(point - center)])
    return view.channels, np.asarray(meta, dtype=np.float32)


def wrist_input(rgb, *, detection_channel, bbox=None, size=224):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected wrist RGB images")
    limits = np.array([rgb.shape[1] - 1, rgb.shape[0] - 1])
    scale = min((size - 1) / limits)
    offset = ((size - 1) - limits * scale) / 2
    affine = np.c_[np.eye(2) * scale, offset]
    resized = cv2.warpAffine(rgb, affine, (size, size), flags=cv2.INTER_LINEAR)
    channels = resized.transpose(2, 0, 1).astype(np.float32) / 255
    if not detection_channel:
        return channels
    mask = np.zeros((size, size), dtype=np.float32)
    if bbox is not None:
        corners = np.asarray(bbox).reshape(2, 2) * limits * scale + offset
        lo, hi = np.ceil(corners[0]).astype(int), np.floor(corners[1]).astype(int)
        mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = 1
    return np.concatenate((channels, mask[None]))
