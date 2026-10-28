"""Reconstruct registered LR infrared / HR visible pairs with WDTFuSR."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from data import AlignedPairs
from models import ModelConfig, WDTFuSR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--infrared', type=Path, required=True)
    parser.add_argument('--visible', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/reconstruction'))
    parser.add_argument('--without-guide', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # Checkpoints include optimizer/RNG state; use a trusted training checkpoint.
    saved = torch.load(args.weights, map_location='cpu', weights_only=False)
    if saved.get('format') != 'wdtfusr-v1':
        raise ValueError('Expected a WDTFuSR checkpoint from train.py.')
    model = WDTFuSR(ModelConfig(**saved['configuration']['model'])).to(args.device).eval()
    model.load_state_dict(saved['model'], strict=True)
    dataset = AlignedPairs(args.infrared, args.visible, model.config.scale)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for sample in tqdm(dataset):
            ir, vi = [sample[key][None].to(args.device) for key in ('infrared', 'visible')]
            if args.without_guide:
                vi = torch.zeros_like(vi)
            result = model(ir, vi)[0, 0].clamp(0, 1)
            array = (result.cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(array).save(args.output / (sample['name'] + '.png'))


if __name__ == '__main__':
    main()
