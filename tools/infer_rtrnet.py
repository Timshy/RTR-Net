import argparse
from pathlib import Path
import sys
import yaml
import torch
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.rtr_net import RTRNet
from datasets.dataset_response import ResponseDataset


def save_prob(path, prob):
    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ResponseDataset(cfg['dataset']['image_dir'], cfg['dataset']['response_dir'])
    model = RTRNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sample in dataset:
            x = torch.cat([sample['image'], sample['response']], dim=0).unsqueeze(0).to(device)
            mask, _ = model(x)
            save_prob(save_dir / f"{sample['stem']}.png", mask.squeeze().cpu().numpy())
    print(f'RTR-Net predictions saved to {save_dir}')


if __name__ == '__main__':
    main()
