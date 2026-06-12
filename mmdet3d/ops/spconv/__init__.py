import torch
import spconv.pytorch as spconv

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                   SparseConvTranspose3d, SparseInverseConv2d,
                   SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.core import SparseConvTensor

def scatter_nd(indices, updates, shape):
    """scatter_nd."""
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

from mmcv.cnn import CONV_LAYERS
CONV_LAYERS.register_module('SparseConv3d', module=SparseConv3d, force=True)
CONV_LAYERS.register_module('SubMConv3d', module=SubMConv3d, force=True)
CONV_LAYERS.register_module('SparseConv2d', module=SparseConv2d, force=True)
CONV_LAYERS.register_module('SubMConv2d', module=SubMConv2d, force=True)
CONV_LAYERS.register_module('SparseInverseConv3d', module=SparseInverseConv3d, force=True)
CONV_LAYERS.register_module('SparseInverseConv2d', module=SparseInverseConv2d, force=True)
CONV_LAYERS.register_module('SparseConvTranspose3d', module=SparseConvTranspose3d, force=True)
CONV_LAYERS.register_module('SparseConvTranspose2d', module=SparseConvTranspose2d, force=True)

__all__ = [
    'SparseConv2d',
    'SparseConv3d',
    'SubMConv2d',
    'SubMConv3d',
    'SparseConvTranspose2d',
    'SparseConvTranspose3d',
    'SparseInverseConv2d',
    'SparseInverseConv3d',
    'SparseModule',
    'SparseSequential',
    'SparseMaxPool2d',
    'SparseMaxPool3d',
    'SparseConvTensor',
    'scatter_nd',
]
