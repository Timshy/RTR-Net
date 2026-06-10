import argparse
from pathlib import Path
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.rtr_net import RTRNet
from models.losses import rtr_loss
from datasets.dataset_response import ResponseDataset


def make_skeleton_batch(mask):
    arr = mask.detach().cpu().numpy()
    out = []
    for m in arr:
        out.append(skeletonize(m[0] > 0.5).astype('float32')[None])
    return torch.tensor(out, dtype=mask.dtype, device=mask.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = ResponseDataset(cfg['data']['train_image_dir'], cfg['data']['train_response_dir'], cfg['data']['train_mask_dir'])
    loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train'].get('num_workers', 4))
    model = RTRNet(cfg['model'].get('in_channels', 4), cfg['model'].get('base_channels', 32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    save_dir = Path(cfg['train']['save_dir']); save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg['train']['epochs']):
        model.train(); total = 0.0
        for sample in loader:
            img = sample['image'].to(device)
            resp = sample['response'].to(device)
            mask_gt = sample['mask'].to(device)
            skel_gt = make_skeleton_batch(mask_gt)
            mask_pred, skel_pred = model(torch.cat([img, resp], dim=1))
            loss = rtr_loss(mask_pred, skel_pred, mask_gt, skel_gt, cfg['model'].get('skel_loss_weight', 0.5))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach().cpu())
        print(f'Epoch {epoch + 1}: loss={total / max(len(loader), 1):.4f}')
        torch.save({'model': model.state_dict()}, save_dir / f'rtrnet_epoch_{epoch + 1}.pth')
    torch.save({'model': model.state_dict()}, save_dir / 'rtrnet.pth')


if __name__ == '__main__':
    main()
