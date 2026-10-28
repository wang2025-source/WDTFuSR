"""WDTFuSR: subband modulation, dense attention and reciprocal visible guidance."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import (Mlp, SwinTransformerBlock, Cross_WindowAttention,
                        window_partition, window_reverse)


@dataclass(frozen=True)
class ModelConfig:
    scale: int = 4
    channels: int = 60
    encoder_groups: int = 2
    fusion_modules: int = 3
    reconstruction_groups: int = 2
    dense_units: int = 5
    window_size: int = 8
    heads: int = 6
    mlp_ratio: float = 2.
    residual_scale: float = .2
    visible_dropout: float = .5

    def __post_init__(self):
        if self.scale not in (2, 4, 6):
            raise ValueError("Choose scale 2, 4, or 6.")
        if min(self.channels, self.encoder_groups, self.fusion_modules,
               self.reconstruction_groups, self.dense_units, self.window_size, self.heads) < 1:
            raise ValueError("Network dimensions must be positive.")
        if self.channels < 2 or self.channels % self.heads or not 0 <= self.visible_dropout <= 1:
            raise ValueError("Invalid attention dimensions or visible dropout probability.")


def haar_dwt(x):
    a, b = x[..., 0::2, 0::2], x[..., 1::2, 0::2]
    c, d = x[..., 0::2, 1::2], x[..., 1::2, 1::2]
    return (a+b+c+d)/2, (-a-b+c+d)/2, (-a+b-c+d)/2, (a-b-c+d)/2


def haar_idwt(ll, horizontal, vertical, diagonal):
    values = torch.stack((ll-horizontal-vertical+diagonal,
                          ll+horizontal-vertical-diagonal,
                          ll-horizontal+vertical-diagonal,
                          ll+horizontal+vertical+diagonal), dim=2) / 2
    # PixelShuffle channel order is top-left, top-right, bottom-left, bottom-right.
    return F.pixel_shuffle(values.flatten(1, 2), 2)


class SDWMB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv2d(1, channels, 3, padding=1)
        self.structural = nn.Sequential(nn.Conv2d(1, 1, 7, padding=3, groups=1),
                                        nn.Conv2d(1, channels, 1))
        self.band_mappings = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
                          nn.Conv2d(channels, channels, 3, padding=1)) for _ in range(4)])
        self.project = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, image):
        local = self.local(image)
        height, width = local.shape[-2:]
        padded = F.pad(local, (0, width % 2, 0, height % 2), mode="replicate")
        mapped = [mapping(band) for mapping, band in zip(self.band_mappings, haar_dwt(padded))]
        frequency = haar_idwt(*mapped)[..., :height, :width]
        return self.project(torch.cat((local, self.structural(image) * frequency), dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 30), 1), nn.ReLU(),
            nn.Conv2d(max(1, channels // 30), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.weights(x)


class RDCAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.channels
        self.scale = config.residual_scale
        self.units = nn.ModuleList()
        for index in range(config.dense_units):
            self.units.append(nn.ModuleDict({
                'stl': SwinTransformerBlock(c * (index + 1), (64, 64), config.heads,
                    window_size=config.window_size,
                    shift_size=0 if index % 2 == 0 else config.window_size // 2,
                    mlp_ratio=config.mlp_ratio),
                'projection': nn.Conv2d(c * (index + 1), c, 1),
                'channel': ChannelAttention(c)}))

    def forward(self, x):
        height, width = x.shape[-2:]
        features = [x]
        for index, unit in enumerate(self.units):
            dense = torch.cat(features, dim=1).flatten(2).transpose(1, 2)
            tokens = unit['stl'](dense, (height, width))
            spatial = tokens.transpose(1, 2).reshape(x.shape[0], -1, height, width)
            feature = unit['channel'](unit['projection'](spatial))
            if index + 1 < len(self.units):
                feature = F.leaky_relu(feature, .2)
            features.append(feature)
        return x + self.scale * features[-1]


class DenseGroups(nn.Module):
    def __init__(self, config, count):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(config.channels * (i + 1), config.channels, 1)
                                         for i in range(count))
        self.groups = nn.ModuleList(RDCAG(config) for _ in range(count))

    def forward(self, x):
        features = [x]
        for projection, group in zip(self.projections, self.groups):
            features.append(group(projection(torch.cat(features, dim=1))))
        return x + features[-1]


class ACFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.channels
        self.window_size = config.window_size
        self.self_attention = nn.ModuleList(SwinTransformerBlock(c, (64, 64), config.heads,
            window_size=config.window_size, shift_size=0, mlp_ratio=config.mlp_ratio) for _ in range(2))
        self.cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(2))
        self.cross_attention = nn.ModuleList(Cross_WindowAttention(c,
            (config.window_size, config.window_size), config.heads) for _ in range(2))
        self.output_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(2))
        self.ffn = nn.ModuleList(Mlp(c, int(c * config.mlp_ratio)) for _ in range(2))

    def forward(self, infrared, visible):
        batch, channels, height, width = infrared.shape
        original = [image.flatten(2).transpose(1, 2) for image in (infrared, visible)]
        streams = [module(tokens, (height, width))
                   for module, tokens in zip(self.self_attention, original)]
        normalized = [norm(tokens).reshape(batch, height, width, channels)
                      for norm, tokens in zip(self.cross_norm, streams)]
        windows = [window_partition(x, self.window_size).reshape(-1, self.window_size ** 2, channels)
                   for x in normalized]
        outputs = []
        for index in range(2):
            response = self.cross_attention[index](windows[index], windows[1-index])
            response = window_reverse(response.reshape(-1, self.window_size, self.window_size, channels),
                                      self.window_size, height, width).reshape(batch, height * width, channels)
            updated = streams[index] + response
            updated = updated + self.ffn[index](self.output_norm[index](updated))
            outputs.append(updated.transpose(1, 2).reshape(batch, channels, height, width))
        return tuple(outputs)


class WDTFuSR(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        config = self.config
        c = config.channels
        self.sdwmb = SDWMB(c)
        self.infrared_stem = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(.2))
        self.visible_stem = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.LeakyReLU(.2))
        self.infrared_encoder = DenseGroups(config, config.encoder_groups)
        self.visible_encoder = DenseGroups(config, config.encoder_groups)
        self.fusion = nn.ModuleList(ACFM(config) for _ in range(config.fusion_modules))
        self.fusion_projection = nn.Conv2d(c * 2, c, 3, padding=1)
        self.reconstruction = DenseGroups(config, config.reconstruction_groups)
        self.output = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(.2),
            nn.Conv2d(c, c // 2, 3, padding=1), nn.LeakyReLU(.2), nn.Conv2d(c // 2, 1, 3, padding=1))

    def forward(self, low_ir, visible, *, infrared_upsampled=False):
        if low_ir.ndim != 4 or visible.ndim != 4 or low_ir.shape[1] != 1 or visible.shape[1] != 3:
            raise ValueError("Expected LR infrared [B,1,h,w] and visible guidance [B,3,H,W].")
        height, width = visible.shape[-2:]
        expected = low_ir.shape[-2:] if infrared_upsampled else (
            low_ir.shape[-2] * self.config.scale, low_ir.shape[-1] * self.config.scale)
        if low_ir.shape[0] != visible.shape[0] or tuple(expected) != (height, width):
            raise ValueError("The registered guidance dimensions must equal LR dimensions times scale.")
        infrared = low_ir if infrared_upsampled else F.interpolate(
            low_ir, size=(height, width), mode="bicubic", align_corners=False)
        if self.training and self.config.visible_dropout:
            keep = visible.new_empty(visible.shape[0], 1, 1, 1).bernoulli_(1-self.config.visible_dropout)
            visible = visible * keep
        ws = self.config.window_size
        padding = (0, -width % ws, 0, -height % ws)
        infrared = F.pad(infrared, padding, mode="replicate")
        visible = F.pad(visible, padding, mode="replicate")
        infrared = self.infrared_encoder(self.infrared_stem(self.sdwmb(infrared)))
        visible = self.visible_encoder(self.visible_stem(visible))
        for module in self.fusion:
            infrared, visible = module(infrared, visible)
        fused = self.fusion_projection(torch.cat((infrared, visible), dim=1))
        return self.output(self.reconstruction(fused))[..., :height, :width]

    def configuration(self):
        return asdict(self.config)
