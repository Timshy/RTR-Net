import cv2
import numpy as np
from skimage.morphology import skeletonize


def binarize(x, threshold=0.5):
    return (x >= threshold).astype(np.uint8)


def dilate(mask, radius):
    k = 2 * int(radius) + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def skeleton(mask):
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8)


def skeleton_connectivity(mask, eps=1e-6):
    skel = skeleton(mask)
    total = float(skel.sum())
    if total <= 0:
        return 0.0
    lcc = largest_connected_component(skel)
    return float(lcc.sum()) / (total + eps)
