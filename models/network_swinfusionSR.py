# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def dwt_init(x):
    
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    
    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))
    
    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    
    x_LL = x1 + x2 + x3 + x4    
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    
    return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)

class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallScaleFeatureExtractor, self).__init__()
        
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv3x3(x)

class LargeScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeScaleFeatureExtractor, self).__init__()
        
        self.dwconv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.dwconv7x7(x)
        return self.pointwise(x)

class WaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(WaveletAttention, self).__init__()
        
        self.dwt = DWT()
        # self.channel_attention = ChannelAttentionModified(in_channels, reduction_ratio)

    def forward(self, x):
        dwt_result = self.dwt(x)
        
        cA, cH, cV, cD = dwt_result

        cA = F.interpolate(cA, scale_factor=2, mode='bicubic', align_corners=False)
        cH = F.interpolate(cH, scale_factor=2, mode='bicubic', align_corners=False)
        cV = F.interpolate(cV, scale_factor=2, mode='bicubic', align_corners=False)
        cD = F.interpolate(cD, scale_factor=2, mode='bicubic', align_corners=False)
          
        return cA, cH, cV, cD

class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapping, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        return x

class FeatureModulation(nn.Module):
    def __init__(self, in_channels, wavelet_channels, scale_factor=1):
        super(FeatureModulation, self).__init__()
        self.mapping = FeatureMapping(in_channels * 4, wavelet_channels * scale_factor)
        self.scale_factor = scale_factor
    
    def forward(self, large_feature_map, detail_feature_map):
        # Generate modulation parameters from detail features
        modulation_params = self.mapping(detail_feature_map)
        
        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor, mode='bilinear')
        
        # Apply modulation to the large scale feature map
        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size, mode='bilinear', align_corners=False)
        modulated_feature_map = large_feature_map * modulation_params
        return modulated_feature_map

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, wavelet_channels):
        super(FeatureFusionModule, self).__init__()
        
        # mid_channels = in_channels // 2
        mid_channels = in_channels
        wavelet_channels = in_channels
        self.small_scale_extractor = SmallScaleFeatureExtractor(in_channels, mid_channels)
        self.large_scale_extractor = LargeScaleFeatureExtractor(in_channels, in_channels)
        
        
        self.wavelet_attention = WaveletAttention(mid_channels)
        
        self.feature_modulation = FeatureModulation(mid_channels, wavelet_channels, scale_factor=1)
        
        
        self.fusion_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)

        # Fix the parameters in this module
        for param in self.parameters():
            param.requires_grad = True # False/True

    def forward(self, x):
        
        small_scale_features = self.small_scale_extractor(x)
        large_scale_features = self.large_scale_extractor(x)
        
        
        
        cA, attn_cH, attn_cV, attn_cD = self.wavelet_attention(small_scale_features)

        recombined_features = torch.cat((cA, attn_cH, attn_cV, attn_cD), dim=1)
        

        modulated_large_scale_features = self.feature_modulation(large_scale_features, recombined_features)
        
     
        combined_features = torch.cat([modulated_large_scale_features, small_scale_features], dim=1)
        # combined_features = torch.cat([large_scale_features, small_scale_features], dim=1)
        fused_features = self.fusion_conv(combined_features)
        
        return fused_features

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn += relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Cross_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class Cross_SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.calculate_mask(x_size).to(y.device))

        # merge windows
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Cross_BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Cross_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    # 以x为主, y只是辅助x的特征提取
    def forward(self, x, y, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, y = checkpoint.checkpoint(blk, x, y, x_size)
            else:
                x, y = blk(x, y, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
            y = self.downsample(y)
        # print("Cross_BasicLayer:", type(x), type(y))
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        # if resi_connection == '1conv':
        #     self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # elif resi_connection == '3conv':
        #     # to save parameters and memory
        #     self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

    def forward(self, x, x_size):
        # return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        # return self.residual_group(x, x_size) + x
        return self.residual_group(x, x_size)

    def flops(self):    
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9

        return flops

class CRSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(CRSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = Cross_BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        self.residual_group_A = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        self.residual_group_B = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        # if resi_connection == '1conv':
        #     self.conv_A = nn.Conv2d(dim, dim, 3, 1, 1)
        #     self.conv_B = nn.Conv2d(dim, dim, 3, 1, 1)
        # elif resi_connection == '3conv':
        #     # to save parameters and memory
        #     self.conv_A = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim, 3, 1, 1))
        #     self.conv_B = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                               nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

    def forward(self, x, y, x_size):
        ## Intra-Modal Fusion
        # x = self.residual_group_A(x, x_size) + x
        # y = self.residual_group_B(y, x_size) + y
        # 不使用残差连接
        x = self.residual_group_A(x, x_size)
        y = self.residual_group_B(y, x_size)
        ## Inter-Modal Fusion
        x1 = x
        y1 = y
        x, y = self.residual_group(x1, y1, x_size)
        # x = self.patch_embed(self.conv_A(self.patch_unembed(x, x_size))) + x1
        # y = self.patch_embed(self.conv_B(self.patch_unembed(y, x_size))) + y1
        # 不使用残差连接
        # x = x + x1
        # y = y + y1
        return x, y

    def flops(self):
        flops = 0
        flops += self.residual_group_A.flops()
        flops += self.residual_group_B.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9

        return flops

class RDG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer,
                 gc, patch_size, img_size, compress_ratio=3, squeeze_factor=30, overlap_ratio=0.5):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = nn.Conv2d(dim, gc, 1)
        self.cab1 = CAB(gc, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc) % num_heads), window_size=window_size,
                                          shift_size=window_size // 2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = nn.Conv2d(dim + gc, gc, 1)
        self.cab2 = CAB(gc, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = nn.Conv2d(dim + gc * 2, gc, 1)
        self.cab3 = CAB(gc, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc) % num_heads), window_size=window_size,
                                          shift_size=window_size // 2,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = nn.Conv2d(dim + gc * 3, gc, 1)
        self.cab4 = CAB(gc, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.swin5 = SwinTransformerBlock(dim + 4 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 4 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)

        self.adjust5 = nn.Conv2d(dim + gc * 4, dim, 1)
        self.cab5 = CAB(dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, xsize):
        x1 = self.pe(self.lrelu(self.cab1(self.adjust1(self.pue(self.swin1(x, xsize), xsize)))))
        x2 = self.pe(self.lrelu(self.cab2(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize)))))
        x3 = self.pe(self.lrelu(self.cab3(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize)))))
        x4 = self.pe(self.lrelu(self.cab4(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize)))))
        x5 = self.pe(self.cab5(self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize))))

        # x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, xsize), xsize))))
        # x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
        # x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
        # x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
        # x5 = self.pe(self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))

        return x5 * 0.2 + x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        # x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinFusionSR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        contrastive: If True, use contrastive learning to train the model. Default: False
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',contrastive=False,
                 **kwargs):
        super(SwinFusionSR, self).__init__()
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.contrastive = contrastive

        #####################################################################################################
        ################################### 1, Wavelet Transform Feature Modulation##########################
        self.featureFusionmodule = FeatureFusionModule(in_chans, num_out_ch, upscale, in_chans)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        ####修改shallow feature extraction 网络, 修改为2个3x3的卷积####
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim_temp, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.Ex_num_layers = len(Ex_depths)
        self.Fusion_num_layers = len(Fusion_depths)
        self.Re_num_layers = len(Re_depths)

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        #contrastive part
        if contrastive:
            self.contrastive_head1= nn.Sequential(
                                       nn.AdaptiveAvgPool2d((3,3)),
                                       nn.Flatten(1,3),
                                       nn.Linear(in_features=self.embed_dim*9, out_features=512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(in_features=512, out_features=64),
                                       nn.BatchNorm1d(64),
                                   )
            self.contrastive_head2= nn.Sequential(
                                       nn.AdaptiveAvgPool2d((3,3)),
                                       nn.Flatten(1,3),
                                       nn.Linear(in_features=self.embed_dim*9, out_features=512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(in_features=512, out_features=64),
                                       nn.BatchNorm1d(64),
                                   )

        # stochastic depth
        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]  # stochastic depth decay rule
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]  # stochastic depth decay rule
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]  # stochastic depth decay rule
        # build Residual Swin Transformer blocks (RSTB)
        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            # layer = RSTB(dim=embed_dim,
            #              input_resolution=(patches_resolution[0],
            #                                patches_resolution[1]),
            #              depth=Ex_depths[i_layer],
            #              num_heads=Ex_num_heads[i_layer],
            #              window_size=window_size,
            #              mlp_ratio=self.mlp_ratio,
            #              qkv_bias=qkv_bias, qk_scale=qk_scale,
            #              drop=drop_rate, attn_drop=attn_drop_rate,
            #              drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
            #              norm_layer=norm_layer,
            #              downsample=None,
            #              use_checkpoint=use_checkpoint,
            #              img_size=img_size,
            #              patch_size=patch_size,
            #              resi_connection=resi_connection
            #              )
            layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=Ex_num_heads[i_layer], window_size=window_size, depth=0,
                        shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                        norm_layer=norm_layer, gc=32, img_size=img_size, patch_size=patch_size,
                        compress_ratio=3, squeeze_factor=30, overlap_ratio=0.5)
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.num_features)

        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            # layer = RSTB(dim=embed_dim,
            #              input_resolution=(patches_resolution[0],
            #                                patches_resolution[1]),
            #              depth=Ex_depths[i_layer],
            #              num_heads=Ex_num_heads[i_layer],
            #              window_size=window_size,
            #              mlp_ratio=self.mlp_ratio,
            #              qkv_bias=qkv_bias, qk_scale=qk_scale,
            #              drop=drop_rate, attn_drop=attn_drop_rate,
            #              drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
            #              norm_layer=norm_layer,
            #              downsample=None,
            #              use_checkpoint=use_checkpoint,
            #              img_size=img_size,
            #              patch_size=patch_size,
            #              resi_connection=resi_connection
            #              )
            layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=Ex_num_heads[i_layer], window_size=window_size, depth=0,
                        shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                        norm_layer=norm_layer, gc=32, img_size=img_size, patch_size=patch_size,
                        compress_ratio=3, squeeze_factor=30, overlap_ratio=0.5)
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.num_features)
        
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.num_features)
        self.norm_Fusion_B = norm_layer(self.num_features)
        
        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            # layer = RSTB(dim=embed_dim,
            #              input_resolution=(patches_resolution[0],
            #                                patches_resolution[1]),
            #              depth=Re_depths[i_layer],
            #              num_heads=Re_num_heads[i_layer],
            #              window_size=window_size,
            #              mlp_ratio=self.mlp_ratio,
            #              qkv_bias=qkv_bias, qk_scale=qk_scale,
            #              drop=drop_rate, attn_drop=attn_drop_rate,
            #              drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],  # no impact on SR results
            #              norm_layer=norm_layer,
            #              downsample=None,
            #              use_checkpoint=use_checkpoint,
            #              img_size=img_size,
            #              patch_size=patch_size,
            #              resi_connection=resi_connection
            #              )
            layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=Re_num_heads[i_layer], window_size=window_size, depth=0,
                        shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],
                        norm_layer=norm_layer, gc=32, img_size=img_size, patch_size=patch_size,
                        compress_ratio=3, squeeze_factor=30, overlap_ratio=0.5)
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            # self.conv_after_body_Ex_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            # self.conv_after_body_Ex_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Fusion = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
            # self.conv_after_body_Re = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
            self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))           
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x += self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_A:
            x = layer(x, x_size)

        x = self.norm_Ex_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_Ex_B(self, x):    
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))      
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x += self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_B:
            x = layer(x, x_size)

        x = self.norm_Ex_B(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward_features_Fusion(self, x, y):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x += self.absolute_pos_embed
            y += self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)
            # y = layer(y, x, x_size)
            

        x = self.norm_Fusion_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)  # B L C
        y = self.patch_unembed(y, x_size)

        if self.contrastive:
            x_head = self.contrastive_head1(x)
            y_head = self.contrastive_head2(y)

        x = torch.cat([x, y], 1)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_after_body_Fusion(x))
        if self.contrastive:
            return x, x_head, y_head
        else:
            return x

    def forward_features_Re(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x, x_size)

        x = self.norm_Re(x)  # B L C
        x = self.patch_unembed(x, x_size)
        # Convolution 
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x

    def forward(self, low_IR, guided):
        # print("Initializing the model")
        x = low_IR
        #print("min:max", torch.min(x), torch.max(x))
        y = guided
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)
        padded_low_IR = x  # Save padded low_IR

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # featureFusionmodule Using
        x = self.featureFusionmodule(x)

        # Feedforward
        x = self.forward_features_Ex_A(x)
        y = self.forward_features_Ex_B(y)
        if self.contrastive:
            x, x_head, y_head = self.forward_features_Fusion(x, y)
        else:  
            x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)      
        #print("min:max", torch.min(x), torch.max(x))
            
        
        x = x / self.img_range + self.mean + padded_low_IR
        if self.contrastive:
            return  x[:, :, :H*self.upscale, :W*self.upscale], x_head, y_head
        else:
            return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_Ex_A):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Ex_B):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Fusion):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Re):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinFusion(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)