import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from qatf.qatf import qatf_fusion


def load_prob(path):
    arr = np.asarray(Image.open(path).convert('L')).astype(np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    return arr


def save_mask(path, mask):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rtr_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    response_dir = Path(cfg['dataset']['response_dir'])
    qcfg = cfg.get('qatf', {})
    rtr_dir = Path(args.rtr_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for rtr_path in sorted(rtr_dir.glob('*')):
        if rtr_path.suffix.lower() not in ['.png', '.jpg', '.tif', '.tiff']:
            continue
        resp_path = None
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            p = response_dir / f'{rtr_path.stem}{ext}'
            if p.exists():
                resp_path = p
                break
        if resp_path is None:
            continue
        final, _ = qatf_fusion(load_prob(resp_path), load_prob(rtr_path), qcfg)
        save_mask(save_dir / f'{rtr_path.stem}.png', final)
    print(f'QATF results saved to {save_dir}')


if __name__ == '__main__':
    main()
