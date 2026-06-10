from pathlib import Path
import torch
from torch.utils.data import Dataset
from .transforms import load_rgb, load_gray, normalize_rgb


class ResponseDataset(Dataset):
    def __init__(self, image_dir, response_dir, mask_dir=None):
        self.image_dir = Path(image_dir)
        self.response_dir = Path(response_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.items = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
        if not self.items:
            raise RuntimeError(f'No images found in {self.image_dir}')

    def __len__(self):
        return len(self.items)

    def _match(self, directory, stem):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            p = directory / f'{stem}{ext}'
            if p.exists():
                return p
        raise FileNotFoundError(f'No file with stem {stem} found in {directory}')

    def __getitem__(self, idx):
        img_path = self.items[idx]
        stem = img_path.stem
        img = normalize_rgb(load_rgb(img_path))
        resp = load_gray(self._match(self.response_dir, stem))[..., None]
        sample = {
            'image': torch.from_numpy(img.transpose(2, 0, 1)).float(),
            'response': torch.from_numpy(resp.transpose(2, 0, 1)).float(),
            'stem': stem,
        }
        if self.mask_dir is not None:
            mask = load_gray(self._match(self.mask_dir, stem))[None, ...]
            sample['mask'] = torch.from_numpy(mask).float()
        return sample
