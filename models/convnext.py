# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Taken from
# https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/models/convnext.py


import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


class FuzzyNextMinBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        kernel_size (int): dws kernel_size
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            warnings.warn(f'Using kernel_size: {kernel_size}')

        self.dwconv_left = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=dim)  # depthwise conv
        self.dwconv_right = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                      padding=kernel_size // 2, groups=dim)
        self.instance_norm_relu = nn.Sequential(
            nn.InstanceNorm2d(dim),
            nn.ReLU()
        )
        self.min = Minimum()

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.norm2 = LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        #self.lbda = .5
        self.lbda = nn.Parameter(torch.tensor(
            [.5], requires_grad=False
        ).float())

    def forward(self, x):
        input = x
        x_conv = self.dwconv_left(x)
        x_right = self.dwconv_right(x)

        x_left = self.instance_norm_relu(x_conv)
        x_right = self.instance_norm_relu(x_right)
        x_min = self.min(x_left, x_right)

        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.norm1(x_conv)

        x_min = x_min.permute(0, 2, 3, 1)
        x_min = self.norm2(x_min)

        if self.training:
            lbda = (torch.rand(1) >= .5).float().to(x.device) + 0. * self.lbda
        else:
            lbda = self.lbda

        x = lbda * x_min + (1. - lbda) * x_conv

        # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NextMinBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        kernel_size (int): dws kernel_size
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            warnings.warn(f'Using kernel_size: {kernel_size}')

        self.dwconv_left = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=dim)  # depthwise conv
        self.dwconv_right = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                      padding=kernel_size // 2, groups=dim)
        self.instance_norm_relu = nn.Sequential(
            nn.InstanceNorm2d(dim),
            nn.ReLU()
        )
        self.min = Minimum()

        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_left = self.dwconv_left(x)
        x_right = self.dwconv_right(x)

        x_left = self.instance_norm_relu(x_left)
        x_right = self.instance_norm_relu(x_right)

        x = self.min(x_left, x_right)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NextMinMinusLambdaBlock(NextMinBlock):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__(
            dim, drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size
        )
        self.lambda_ = 2.

    def forward(self, x):
        input = x
        x_left = self.dwconv_left(x)
        x_right = self.dwconv_right(x)

        x_left = self.instance_norm_relu(x_left)
        x_right = self.instance_norm_relu(x_right)

        x = self.lambda_ * self.min(x_left, x_right) - x_left

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NextMinMinusAbsBlockNoNorm(NextMinBlock):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__(
            dim, drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size
        )
        self.lambda_ = 2.
        self.abs = Abs()
        self.instance_norm_relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x):
        input = x
        x_left = self.dwconv_left(x)
        x_right = self.dwconv_right(x)

        x_left = self.instance_norm_relu(x_left)
        x_right = self.instance_norm_relu(x_right)

        x = (
            self.lambda_ * self.min(x_left, x_right) -
            self.abs(x_left - x_right)
        )

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NextMinMinusLambdaBlockBN(NextMinMinusLambdaBlock):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__(
            dim, drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size
        )
        self.instance_norm_relu = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )


class NextMinMinusLambdaBlockNoNorm(NextMinMinusLambdaBlock):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__(
            dim, drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size
        )
        self.instance_norm_relu = nn.Sequential(
            nn.ReLU()
        )


class Minimum(nn.Module):
    def forward(self, x, y):
        # Computes the element-wise minimum of x and y
        return torch.minimum(x, y)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        kernel_size (int): dws kernel_size
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            warnings.warn(f'Using kernel_size: {kernel_size}')

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        strides (tuple(int)): which strides are used for the downsampling layers
        downsample_padding (bool): Use padding for downsampling layers. Default: False
        kernel_size (int): kernel size for the dws-conv. Default: 7
        bitstring (list(int)): whether to substitute the basic block with an Min-block (if bit==1)
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 strides=[4, 2, 2, 2], downsample_padding=False,
                 kernel_size=7,
                 bitstring=None, alternate_block=None
                 ):
        super().__init__()

        if bitstring is not None:
            # remove any space chars
            bitstring = [int(b) for b in bitstring.replace(' ', '')]

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0],
                      kernel_size=(5 if downsample_padding else 4),
                      stride=strides[0],
                      padding=(2 if downsample_padding else 0)
                      ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first",
                      )
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1],
                          kernel_size=(3 if downsample_padding else 2),
                          stride=strides[i + 1],
                          padding=(1 if downsample_padding else 0)),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[self._get_block(cur + j, bitstring, alternate_block)(
                    dim=dims[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=kernel_size
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _get_block(self, index, bitstring, alternate_block):
        if bitstring is None:
            return Block

        if alternate_block is not None:
            new_block = {
                'fuzzy': FuzzyNextMinBlock,
                'minuslambda': NextMinMinusLambdaBlock,
                'minuslambdaBN': NextMinMinusLambdaBlockBN,
                'minuslambdaNoNorm': NextMinMinusLambdaBlockNoNorm,
                'minusAbsNoNorm': NextMinMinusAbsBlockNoNorm,
            }[alternate_block]
            print(f"{index} – Using alternate-block: {alternate_block}")
            return new_block if bitstring[index] == 1 else Block

        return NextMinBlock if bitstring[index] == 1 else Block

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
