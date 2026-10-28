"""Registered infrared/visible pairs and synchronized target-grid augmentation."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in
             {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
    mapping = {p.stem: p for p in paths}
    if len(mapping) != len(paths):
        raise ValueError(f'Duplicate image stems: {folder}')
    if not mapping:
        raise ValueError(f'No images: {folder}')
    return mapping


def read_image(path, mode):
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32) / 255.
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class AlignedPairs(Dataset):
    def __init__(self, infrared, visible, scale, target=None, crop_size=None):
        self.ir, self.vi = images(infrared), images(visible)
        self.gt = images(target) if target else None
        self.names = sorted(self.ir)
        if set(self.ir) != set(self.vi) or (self.gt is not None and set(self.ir) != set(self.gt)):
            raise ValueError('All modalities must contain the same image stems.')
        self.scale, self.crop_size = scale, crop_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        ir, vi = read_image(self.ir[name], 'L'), read_image(self.vi[name], 'RGB')
        if (ir.shape[-2] * self.scale, ir.shape[-1] * self.scale) != vi.shape[-2:]:
            raise ValueError(f'LR/guidance dimensions do not match scale: {name}')
        sample = {'infrared': ir, 'visible': vi, 'name': name}
        if self.gt is not None:
            sample['target'] = read_image(self.gt[name], 'L')
            if sample['target'].shape[-2:] != vi.shape[-2:]:
                raise ValueError(f'Target/guidance dimensions differ: {name}')
        if self.crop_size:
            sample['infrared'] = F.interpolate(ir[None], size=vi.shape[-2:],
                                               mode='bicubic', align_corners=False)[0]
            size = self.crop_size
            h, w = vi.shape[-2:]
            if min(h, w) < size:
                raise ValueError(f'Image smaller than the training patch: {name}')
            top, left = random.randint(0, h-size), random.randint(0, w-size)
            flip_h, flip_v, rotation = random.random() < .5, random.random() < .5, random.randrange(4)
            for key in ('infrared', 'visible', 'target'):
                if key in sample:
                    value = sample[key][..., top:top+size, left:left+size]
                    if flip_h:
                        value = value.flip(-1)
                    if flip_v:
                        value = value.flip(-2)
                    sample[key] = torch.rot90(value, rotation, (-2, -1)).contiguous()
        return sample
