# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (FUSION_LAYERS, MIDDLE_ENCODERS, VOXEL_ENCODERS,
                      build_backbone, build_detector, build_fusion_layer,
                      build_head, build_loss, build_middle_encoder,
                      build_model, build_neck, build_roi_extractor,
                      build_shared_head, build_voxel_encoder)
from .decode_heads import *  # noqa: F401,F403
from .dense_heads import *  # noqa: F401,F403
from .detectors import *  # noqa: F401,F403
from .fusion_layers import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .middle_encoders import *  # noqa: F401,F403
from .model_utils import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .voxel_encoders import *  # noqa: F401,F403

__all__ = [
    'VOXEL_ENCODERS', 'MIDDLE_ENCODERS', 'FUSION_LAYERS', 'build_backbone',
    'build_neck', 'build_roi_extractor', 'build_shared_head', 'build_head',
    'build_loss', 'build_detector', 'build_fusion_layer', 'build_model',
    'build_middle_encoder', 'build_voxel_encoder'
]
# --- CUSTOM PLUGIN AUTOMATIC IMPORTS ---
import sys
import os
import importlib

try:
    import projects
    
    # 1. Register projects paths
    pillarnest_proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../projects/adv_aug/PillarNeSt/projects'))
    focalformer_proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../projects/adv_aug/FocalFormer3D/projects'))
    
    if os.path.exists(pillarnest_proj_path) and pillarnest_proj_path not in projects.__path__:
        projects.__path__.append(pillarnest_proj_path)
    if os.path.exists(focalformer_proj_path) and focalformer_proj_path not in projects.__path__:
        projects.__path__.append(focalformer_proj_path)
        
    # Import base plugin module
    import projects.mmdet3d_plugin
    
    # Pre-add FocalFormer3D's plugin path
    focalformer_plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../projects/adv_aug/FocalFormer3D/projects/mmdet3d_plugin'))
    if os.path.exists(focalformer_plugin_path) and focalformer_plugin_path not in projects.mmdet3d_plugin.__path__:
        projects.mmdet3d_plugin.__path__.append(focalformer_plugin_path)
        
    # Recursive path-merger for namespace package simulation
    def merge_plugin_paths(base_package_name, paths):
        subpackages = set()
        for base_path in paths:
            if not os.path.isdir(base_path):
                continue
            for root, dirs, files in os.walk(base_path):
                if '__init__.py' in files:
                    rel_path = os.path.relpath(root, base_path)
                    if rel_path == '.':
                        continue
                    subpackages.add(rel_path)
        sorted_subpackages = sorted(list(subpackages), key=lambda x: x.count(os.sep))
        for rel_path in sorted_subpackages:
            parts = rel_path.split(os.sep)
            pkg_name = f"{base_package_name}." + ".".join(parts)
            pkg_paths = []
            for base_path in paths:
                p = os.path.abspath(os.path.join(base_path, rel_path))
                if os.path.isdir(p) and p not in pkg_paths:
                    pkg_paths.append(p)
            try:
                mod = importlib.import_module(pkg_name)
                for p in pkg_paths:
                    if p not in mod.__path__:
                        mod.__path__.append(p)
            except Exception:
                if pkg_name in sys.modules:
                    mod = sys.modules[pkg_name]
                    if hasattr(mod, '__path__'):
                        for p in pkg_paths:
                            if p not in mod.__path__:
                                mod.__path__.append(p)

    pillarnest_plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../projects/adv_aug/PillarNeSt/projects/mmdet3d_plugin'))
    merge_plugin_paths('projects.mmdet3d_plugin', [pillarnest_plugin_path, focalformer_plugin_path])
    
    # 2. Import PillarNeSt plugin modules to register them
    import projects.mmdet3d_plugin.models.backbones.convnext_pc
    import projects.mmdet3d_plugin.models.dense_heads.centerpoint_plus_head
    import projects.mmdet3d_plugin.models.voxel_encoders.pillar_encoder
    import projects.mmdet3d_plugin.core.bbox.coder.centerpoint_bbox_coders

    # 3. Import FocalFormer3D plugin modules to register them
    import projects.mmdet3d_plugin.models.necks.focal_encoder
    import projects.mmdet3d_plugin.models.dense_heads.focal_decoder
    import projects.mmdet3d_plugin.models.detectors.focalformer3d
    import projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner
    import projects.mmdet3d_plugin.core.bbox.coders.transfusion_bbox_coder
    import projects.mmdet3d_plugin.core.hook.fading
    import projects.mmdet3d_plugin.datasets.pipelines.transform_3d

except Exception as e:
    print(f"Error importing custom plugins: {e}", file=sys.stderr)

