import math
from functools import partial
from typing import List, Optional, Union

import timm
import torch
from timm.models import layers, resnet
from timm.models.helpers import build_model_with_cfg
from torch import nn
import torchvision
from .utils import (
    init_parameters,
    DEFAULT_WEIGHT_INIT,
    _infer_common_length,
    _maybe_expand_list,
)
class CNNEncoder(nn.Sequential):
    """Simple convolutional encoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = "default",
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.get_same_padding(kernel_size, stride),
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Union[str, int]:
        """Try to infer same padding for convolutions."""
        # This method is very lazily implemented, but oh well..
        if stride == 1:
            return "same"
        if kernel_size == 3:
            if stride == 2:
                return 1
        elif kernel_size == 5:
            if stride == 2:
                return 2

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")

def make_slot_attention_encoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    downsamplings: int = 0,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNEncoder:
    """CNN encoder as used in Slot Attention paper.

    By default, 4 layers with 64 channels each, keeping the spatial input resolution the same.

    This encoder is also used by SAVi, in the following configurations:

    - for image resolution 64: feature_multiplier=0.5, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1

    and STEVE, in the following configurations:

    - for image resolution 64: feature_multiplier=1, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1
    """
    assert 0 <= downsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * downsamplings + [1] * (4 - downsamplings)
    return CNNEncoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )


def _create_savi_resnet(block, stages, **kwargs) -> resnet.ResNet:
    """ResNet as used by SAVi and SAVi++, adapted from SAVi codebase.

    The differences to the normal timm ResNet implementation are to use group norm instead of batch
    norm, and to use 3x3 filters, 1x1 strides and no max pooling in the stem.

    Returns 16x16 feature maps for input size of 128x128, and 28x28 features maps for inputs of
    size 224x224.
    """
    model_args = dict(block=block, layers=stages, norm_layer=layers.GroupNorm, **kwargs)
    model = resnet._create_resnet("resnet34", pretrained=False, **model_args)
    model.conv1 = nn.Conv2d(
        model.conv1.in_channels,
        model.conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.global_pool = nn.Identity()
    model.fc = nn.Identity()
    model.init_weights(zero_init_last=True)  # Re-init weights because we added a new conv layer
    return model

class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out

@timm.models.register_model
def resnet18_savi(pretrained=False, **kwargs):
    """ResNet18 as implemented in SAVi codebase.

    Final features have 512 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet18_savi`.")
    return _create_savi_resnet(resnet.BasicBlock, stages=[2, 2, 2, 2], **kwargs)


@timm.models.register_model
def resnet34_savi(pretrained=False, **kwargs):
    """ResNet34 as used in SAVi and SAVi++ papers.

    Final features have 512 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet34_savi`.")
    return _create_savi_resnet(resnet.BasicBlock, stages=[3, 4, 6, 3], **kwargs)


@timm.models.register_model
def resnet50_savi(pretrained=False, **kwargs):
    """ResNet50 as implemented in SAVi codebase.

    Final features have 2048 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet50_savi`.")
    return _create_savi_resnet(resnet.Bottleneck, stages=[3, 4, 6, 3], **kwargs)

def _resnet50_dino_pretrained_filter(state_dict, model):
    del model.fc
    return state_dict


@timm.models.register_model
def resnet50_dino(pretrained=False, **kwargs):
    """ResNet50 pre-trained with DINO, without classification head.

    Weights from https://github.com/facebookresearch/dino
    """
    kwargs["pretrained_cfg"] = resnet._cfg(
        url=(
            "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/"
            "dino_resnet50_pretrain.pth"
        )
    )
    model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    model = build_model_with_cfg(
        resnet.ResNet,
        "resnet50_dino",
        pretrained=pretrained,
        pretrained_filter_fn=_resnet50_dino_pretrained_filter,
        **model_args,
    )
    return model