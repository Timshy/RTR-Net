import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def load_mask(path):
    arr = np.asarray(Image.open(path).convert('L'))
    return (arr > 127).astype(np.uint8)


def cldice(pred, gt, eps=1e-6):
    skel_pred = skeletonize(pred.astype(bool)).astype(np.uint8)
    skel_gt = skeletonize(gt.astype(bool)).astype(np.uint8)
    tprec = (skel_pred & gt).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt & pred).sum() / (skel_gt.sum() + eps)
    score = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return float(score), float(tprec), float(tsens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    scores, precs, sens = [], [], []
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
            s, p, r = cldice(load_mask(pred_path), load_mask(gt_path))
            scores.append(s); precs.append(p); sens.append(r)
    if scores:
        print(f'Mean clDice: {np.mean(scores) * 100:.2f}')
        print(f'Topology precision: {np.mean(precs) * 100:.2f}')
        print(f'Topology sensitivity: {np.mean(sens) * 100:.2f}')
    else:
        print('No matched files found.')


if __name__ == '__main__':
    main()
