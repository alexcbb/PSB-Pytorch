import torch
import torch.nn as nn
import einops
from typing import Tuple, Dict, Optional, Union, List
from .utils import (
    init_parameters,
    DEFAULT_WEIGHT_INIT,
    _infer_common_length,
    _maybe_expand_list,
)
from .layers import CoordinatePositionEmbed

class CNNDecoder(nn.Sequential):
    """Simple convolutional decoder.

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
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            padding, output_padding = self.get_same_padding(kernel_size, stride)
            layers.append(
                nn.ConvTranspose2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
        """Try to infer same padding for transposed convolutions."""
        # This method is very lazily implemented, but oh well..
        if kernel_size == 3:
            if stride == 1:
                return 1, 0
            if stride == 2:
                return 1, 1
        elif kernel_size == 5:
            if stride == 1:
                return 2, 0
            if stride == 2:
                return 2, 1

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")

def make_savi_decoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    upsamplings: int = 4,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNDecoder:
    """CNN encoder as used in SAVi paper.

    By default, 4 layers with 64 channels each, upscaling from a 8x8 feature map to 128x128.
    """
    assert 0 <= upsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * upsamplings + [1] * (4 - upsamplings)
    return CNNDecoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )

class SpatialBroadcastDecoder(nn.Module):
    """Decoder that reconstructs a spatial map independently per slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        backbone: nn.Module,
        initial_size: Union[int, Tuple[int, int]] = 8,
        backbone_dim: Optional[int] = None,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        if isinstance(initial_size, int):
            self.initial_size = (initial_size, initial_size)
        else:
            self.initial_size = initial_size

        if pos_embed is None:
            self.pos_embed = CoordinatePositionEmbed(inp_dim, initial_size)
        else:
            self.pos_embed = pos_embed

        self.backbone = backbone

        if output_transform is None:
            if backbone_dim is None:
                raise ValueError("Need to provide backbone dim if output_transform is unspecified")
            self.output_transform = nn.Conv2d(backbone_dim, outp_dim + 1, 1, 1)
        else:
            self.output_transform = output_transform

        self.init_parameters()

    def init_parameters(self):
        if isinstance(self.output_transform, nn.Conv2d):
            init_parameters(self.output_transform)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, _ = slots.shape
        slots = einops.repeat(
            slots, "b s d -> (b s) d h w", h=self.initial_size[0], w=self.initial_size[1]
        )
        slots = self.pos_embed(slots)
        features = self.backbone(slots)
        outputs = self.output_transform(features)
        outputs = einops.rearrange(outputs, "(b s) ... -> b s ...", b=bs, s=n_slots)
        recons, alpha = einops.unpack(outputs, [[self.outp_dim], [1]], "b s * h w")

        masks = torch.softmax(alpha, dim=1)
        recon_combined = torch.sum(masks*recons, dim=1)

        return {"reconstruction": recons, "recon_combined": recon_combined, "masks": masks.squeeze(2)}
