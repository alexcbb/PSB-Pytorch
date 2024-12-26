import torch
import torch.nn as nn
from typing import Iterable, Union, List

DEFAULT_WEIGHT_INIT = "default"

def _infer_common_length(fail_on_missing_length=True, **kwargs) -> int:
    """Given kwargs of scalars and lists, checks that all lists have the same length and returns it.

    Optionally fails if no length was provided.
    """
    length = None
    name = None
    for cur_name, arg in kwargs.items():
        if isinstance(arg, (tuple, list)):
            cur_length = len(arg)
            if length is None:
                length = cur_length
                name = cur_name
            elif cur_length != length:
                raise ValueError(
                    f"Inconsistent lengths: {cur_name} has length {cur_length}, "
                    f"but {name} has length {length}"
                )

    if fail_on_missing_length and length is None:
        names = ", ".join(f"`{key}`" for key in kwargs.keys())
        raise ValueError(f"Need to specify a list for at least one of {names}.")

    return length

def _maybe_expand_list(arg: Union[int, List[int]], length: int) -> list:
    if not isinstance(arg, (tuple, list)):
        return [arg] * length

    return list(arg)

def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)

def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')

def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')

def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """ConvTranspose - Norm - Act."""
    deconv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)

def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    if isinstance(in_size, int):
        return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1
    elif isinstance(in_size, (tuple, list)):
        return type(in_size)((deconv_out_shape(s, stride, padding, kernel_size,
                                               out_padding, dilation)
                              for s in in_size))
    else:
        raise TypeError(f'Got invalid type {type(in_size)} for `in_size`')

def get_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.Conv{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)
