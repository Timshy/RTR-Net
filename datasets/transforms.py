import numpy as np
from PIL import Image


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def load_gray(path):
    arr = np.asarray(Image.open(path).convert('L'))
    if arr.max() > 1:
        arr = arr / 255.0
    return arr.astype(np.float32)


def normalize_rgb(img):
    return img.astype(np.float32) / 255.0
