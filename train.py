"""Train WDTFuSR with synchronized target-grid patches and pixel-wise L1 loss."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AlignedPairs
from models import ModelConfig, WDTFuSR


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('options/train_x4.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/wdtfusr'))
    parser.add_argument('--seed', type=int, default=60)
    parser.add_argument('--guide-dropout', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.guide_dropout is not None:
        config['model']['visible_dropout'] = args.guide_dropout
    device = torch.device(args.device)
    model = WDTFuSR(ModelConfig(**config['model'])).to(device)
    protocol = config['training']
    if protocol['epochs'] < 1 or protocol['batch_size'] < 1 or protocol['crop_size'] < 1:
        raise ValueError('Training dimensions and epochs must be positive.')
    dataset = AlignedPairs(**config['data'], scale=model.config.scale, crop_size=protocol['crop_size'])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=protocol['batch_size'], shuffle=True,
                        num_workers=protocol['workers'], worker_init_fn=seed_worker, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=protocol['decay_epochs'], gamma=.5)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get('format') != 'wdtfusr-v1' or saved['configuration'] != config:
            raise ValueError('Checkpoint format or training configuration differs.')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start = saved['epoch']
        random.setstate(saved['random_state'])
        np.random.set_state(saved['numpy_state'])
        torch.set_rng_state(saved['torch_state'].cpu())
        generator.set_state(saved['loader_state'].cpu())
        if torch.cuda.is_available() and saved['cuda_state'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_state']])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'last.pth').exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume.')
    for epoch in range(start, protocol['epochs']):
        model.train()
        total, count = 0., 0
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{protocol["epochs"]}'):
            ir, vi, target = [batch[key].to(device) for key in ('infrared', 'visible', 'target')]
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(ir, vi, infrared_upsampled=True), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total += loss.item() * ir.shape[0]
            count += ir.shape[0]
        scheduler.step()
        state = dict(format='wdtfusr-v1', model=model.state_dict(), configuration=config,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch+1,
            random_state=random.getstate(), numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            loader_state=generator.get_state())
        temporary = args.output / 'last.tmp'
        torch.save(state, temporary)
        temporary.replace(args.output / 'last.pth')
        if (epoch+1) % 100 == 0:
            torch.save(state, args.output / f'epoch_{epoch+1:04}.pth')
        print(f'Epoch {epoch+1}: L1={total/count:.6f}')


if __name__ == '__main__':
    main()
