# Copyright (c) OpenMMLab. All rights reserved.

from mmdet3d.models.dense_heads import CenterHead
from .centerpoint_plus_head import CenterPlusHead


__all__ = [
    'CenterHead', 'CenterPlusHead'
]
