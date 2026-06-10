import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load_mask(path):
    arr = np.asarray(Image.open(path).convert('L'))
    return (arr > 127).astype(np.uint8)


def iou(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / (float(union) + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    scores = []
    for pred_path in sorted(pred_dir.glob('*')):
        if pred_path.suffix.lower() not in ['.png', '.jpg', '.tif', '.tiff']:
            continue
        gt_path = None
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            p = gt_dir / f'{pred_path.stem}{ext}'
            if p.exists():
                gt_path = p
                break
        if gt_path is not None:
            scores.append(iou(load_mask(pred_path), load_mask(gt_path)))
    print(f'Mean IoU: {np.mean(scores) * 100:.2f}' if scores else 'No matched files found.')


if __name__ == '__main__':
    main()
