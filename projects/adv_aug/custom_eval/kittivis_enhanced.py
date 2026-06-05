import argparse
import os
import numpy as np
import matplotlib
# Force headless backend for SLURM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet3d.registry import MODELS

# --- Visualization Constants ---
AXES_LIMITS = [
    [-5, 80],   # X axis range
    [-45, 45],  # Y axis range
    [-4, 2]     # Z axis range
]
AXES_STR = ['X', 'Y', 'Z']

# KITTI 3-Class Default
CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car'] 
LABEL_MAP = np.array(CLASS_NAMES)

# Enhanced bright colors for dark background
BOX_COLORS = {
    'Car': '#00ff88',        # Bright green
    'Pedestrian': '#ff6b6b', # Coral red
    'Cyclist': '#4ecdc4'     # Teal
}

# Original colors (for light background fallback)
COLORS_LIGHT = {
    'Car': 'b',
    'Pedestrian': 'y',
    'Cyclist': 'g'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Adversarial Point Clouds')
    parser.add_argument('--cfg', type=str, required=True, help='Path to mmdet3d config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root path to data (containing training/velodyne)')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to .pkl annotation file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save images')
    parser.add_argument('--frequency', type=int, default=20, help='Visualize every Nth frame')
    parser.add_argument('--show-score-thr', type=float, default=0.3, help='Score threshold for visualizing predictions')
    parser.add_argument('--color-mode', type=str, default='depth', 
                        choices=['depth', 'height', 'intensity', 'density'],
                        help='Point coloring mode: depth, height, intensity, or density')
    parser.add_argument('--dark-theme', action='store_true', default=True,
                        help='Use dark background theme (better for adversarial visualization)')
    parser.add_argument('--light-theme', action='store_true',
                        help='Use light background theme')
    parser.add_argument('--single', action='store_true',
                        help='Visualize only the first point cloud and exit')
    parser.add_argument('--points-keep-ratio', type=float, default=1.0,
                        help='Ratio of points to display (1.0 = all points, 0.5 = half)')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Size of points in visualization (default: 0.5)')
    parser.add_argument('--no-3d', action='store_true',
                    help='Skip 3D visualization output')
    parser.add_argument('--no-bev', action='store_true',
                        help='Skip BEV (Bird\'s Eye View) output')
    parser.add_argument('--bev-dpi', type=int, default=300,
                        help='DPI for BEV output (default: 300, max recommended: 600)')
    parser.add_argument('--bev-figsize', type=float, nargs=2, default=[24, 20],
                        help='Figure size for BEV in inches (width height)')
    args = parser.parse_args()
    
    # Handle theme flags
    if args.light_theme:
        args.dark_theme = False
    
    return args


# --- Point Cloud Coloring Functions ---

def get_point_colors_by_depth(points, cmap='plasma'):
    """Color points by distance from sensor origin."""
    distances = np.linalg.norm(points[:, :3], axis=1)
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(distances)), distances, 'Distance (m)'


def get_point_colors_by_height(points, cmap='viridis'):
    """Color points by Z-height - good for seeing ground vs objects."""
    z_vals = points[:, 2]
    norm = Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(z_vals)), z_vals, 'Height (m)'


def get_point_colors_by_intensity(points, cmap='hot'):
    """Color points by intensity (4th channel if available)."""
    if points.shape[1] >= 4:
        intensity = points[:, 3]
    else:
        intensity = np.linalg.norm(points[:, :3], axis=1)  # fallback to depth
    norm = Normalize(vmin=intensity.min(), vmax=intensity.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(intensity)), intensity, 'Intensity'


def get_point_colors_by_density(points, radius=0.5, cmap='coolwarm'):
    """Color by local point density - highlights adversarial clusters."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :3])
    density = np.array([len(tree.query_ball_point(p, radius)) for p in points[:, :3]])
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(density)), density, 'Local Density'


def get_point_colors(points, mode='depth'):
    """Get colors based on selected mode."""
    color_funcs = {
        'depth': lambda p: get_point_colors_by_depth(p, 'plasma'),
        'height': lambda p: get_point_colors_by_height(p, 'viridis'),
        'intensity': lambda p: get_point_colors_by_intensity(p, 'hot'),
        'density': lambda p: get_point_colors_by_density(p, 0.5, 'coolwarm'),
    }
    return color_funcs.get(mode, color_funcs['depth'])(points)


# --- Drawing Functions ---

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black', linestyle='-', linewidth=1.5):
    """
    Draws a bounding 3D box in a pyplot axis.
    vertices: (3, 8) numpy array
    """
    vertices = vertices[axes, :] 
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connections
    ] 
    
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=linewidth, linestyle=linestyle)


def setup_dark_theme(fig, ax, is_3d=True):
    """Apply dark theme styling to figure and axes."""
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Style axes
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')
    
    if is_3d:
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='z', colors='white')
        # Make panes transparent/dark
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.grid(True, alpha=0.3)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        ax.grid(True, alpha=0.3, color='gray')


def add_colorbar(fig, ax, cmap, values, label, dark_theme=True):
    """Add a colorbar to the figure."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=values.min(), vmax=values.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label, fontsize=10)
    
    if dark_theme:
        cbar.set_label(label, color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


def display_frame_statistics(point_cloud, gt_boxes_corners, pred_boxes_corners, 
                             gt_labels_numeric, pred_labels_numeric, 
                             output_dir, filename_stem, 
                             color_mode='depth', dark_theme=True,
                             points_keep_ratio=1.0, point_size=0.25,
                             output_3d=True, output_bev=True,
                             bev_dpi=300, bev_figsize=(24, 20)):
    """
    Saves enhanced visualizations comparing ground truth and predicted boxes.
    
    Args:
        output_3d: Whether to output 3D visualization
        output_bev: Whether to output BEV (XY) projection
        bev_dpi: DPI for BEV output (higher = better quality)
        bev_figsize: Figure size for BEV output (width, height in inches)
    """
    # 1. Convert labels
    try:
        gt_labels_string = LABEL_MAP[gt_labels_numeric]
    except IndexError:
        gt_labels_string = []
        
    try:
        pred_labels_string = LABEL_MAP[pred_labels_numeric]
    except IndexError:
        pred_labels_string = []

    # 2. Subsample points
    points_step = int(1. / points_keep_ratio) if points_keep_ratio > 0 else 1
    velo_range = range(0, point_cloud.shape[0], points_step)
    velo_frame = point_cloud[velo_range, :]

    # 3. Get colors based on mode
    print(f"  Computing {color_mode} colors...")
    colors, color_values, color_label = get_point_colors(velo_frame, mode=color_mode)
    
    cmap_names = {
        'depth': 'plasma',
        'height': 'viridis', 
        'intensity': 'hot',
        'density': 'coolwarm'
    }
    cmap_name = cmap_names.get(color_mode, 'plasma')

    box_colors = BOX_COLORS if dark_theme else COLORS_LIGHT

    # --- Create Legend ---
    if dark_theme:
        legend_elements = [
            Line2D([0], [0], color=BOX_COLORS['Car'], lw=2.5, linestyle='-', label='Car (GT)'),
            Line2D([0], [0], color=BOX_COLORS['Car'], lw=2, linestyle='--', label='Car (Pred)'),
            Line2D([0], [0], color=BOX_COLORS['Pedestrian'], lw=2.5, linestyle='-', label='Pedestrian (GT)'),
            Line2D([0], [0], color=BOX_COLORS['Pedestrian'], lw=2, linestyle='--', label='Pedestrian (Pred)'),
            Line2D([0], [0], color=BOX_COLORS['Cyclist'], lw=2.5, linestyle='-', label='Cyclist (GT)'),
            Line2D([0], [0], color=BOX_COLORS['Cyclist'], lw=2, linestyle='--', label='Cyclist (Pred)'),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='-', label='Ground Truth'),
            Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Prediction'),
        ]

    # --- Plot 1: 3D View (Optional) ---
    if output_3d:
        fig_3d = plt.figure(figsize=(16, 12))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.view_init(elev=30, azim=-45)
        
        if dark_theme:
            fig_3d.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig_3d, ax_3d, is_3d=True)
        
        ax_3d.scatter(
            velo_frame[:, 0], velo_frame[:, 1], velo_frame[:, 2],
            c=colors, s=point_size, alpha=0.8
        )
        
        ax_3d.set_title(f'3D LiDAR Scan: {filename_stem} | Color: {color_mode}', fontsize=12)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_xlim3d(*AXES_LIMITS[0])
        ax_3d.set_ylim3d(*AXES_LIMITS[1])
        ax_3d.set_zlim3d(*AXES_LIMITS[2])
        
        for corners, label in zip(gt_boxes_corners, gt_labels_string):
            if label in box_colors:
                draw_box(ax_3d, corners.T, axes=[0, 1, 2], color=box_colors[label], 
                        linestyle='-', linewidth=2.0)
        
        for corners, label in zip(pred_boxes_corners, pred_labels_string):
            if label in box_colors:
                draw_box(ax_3d, corners.T, axes=[0, 1, 2], color=box_colors[label], 
                        linestyle='--', linewidth=1.5)
        
        add_colorbar(fig_3d, ax_3d, cmap_name, color_values, color_label, dark_theme)
        
        legend = ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=9)
        if dark_theme:
            legend.get_frame().set_facecolor('#2d2d44')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        output_path_3d = os.path.join(output_dir, f'{filename_stem}_3d_{color_mode}.png')
        fig_3d.tight_layout()
        plt.savefig(output_path_3d, dpi=150, facecolor=fig_3d.get_facecolor(), bbox_inches='tight')
        plt.close(fig_3d)
        print(f"  Saved: {output_path_3d}")

    # --- Plot 2: High-Resolution BEV (XY) Projection ---
    if output_bev:
        fig_bev = plt.figure(figsize=bev_figsize, dpi=bev_dpi)
        ax_bev = fig_bev.add_subplot(111)
        
        if dark_theme:
            fig_bev.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig_bev, ax_bev, is_3d=False)
        
        # Scatter points
        scatter = ax_bev.scatter(
            velo_frame[:, 0],
            velo_frame[:, 1],
            c=colors,
            s=point_size,
            alpha=0.8
        )
        
        ax_bev.set_title(f'Bird\'s Eye View: {filename_stem} | Color: {color_mode}', 
                         fontsize=18, pad=15)
        ax_bev.set_xlabel('X (m)', fontsize=14)
        ax_bev.set_ylabel('Y (m)', fontsize=14)
        ax_bev.set_xlim(*AXES_LIMITS[0])
        ax_bev.set_ylim(*AXES_LIMITS[1])
        ax_bev.set_aspect('equal')
        ax_bev.tick_params(labelsize=12)
        
        # Draw Ground Truth boxes (Solid, thick)
        for corners, label in zip(gt_boxes_corners, gt_labels_string):
            if label in box_colors:
                draw_box(ax_bev, corners.T, axes=[0, 1], color=box_colors[label], 
                        linestyle='-', linewidth=3.0)
        
        # Draw Prediction boxes (Dashed)
        for corners, label in zip(pred_boxes_corners, pred_labels_string):
            if label in box_colors:
                draw_box(ax_bev, corners.T, axes=[0, 1], color=box_colors[label], 
                        linestyle='--', linewidth=2.5)
        
        # Add colorbar
        add_colorbar(fig_bev, ax_bev, cmap_name, color_values, color_label, dark_theme)
        
        # Legend
        legend = ax_bev.legend(handles=legend_elements, loc='upper right', fontsize=12)
        if dark_theme:
            legend.get_frame().set_facecolor('#2d2d44')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        output_path_bev = os.path.join(output_dir, f'{filename_stem}_bev_{color_mode}.png')
        fig_bev.tight_layout()
        plt.savefig(output_path_bev, dpi=bev_dpi, facecolor=fig_bev.get_facecolor(), bbox_inches='tight')
        plt.close(fig_bev)
        print(f"  Saved: {output_path_bev}")

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load Config
    cfg = Config.fromfile(args.cfg)
    
    # 2. Patch Config with Arguments
    if 'data_root' in cfg:
        cfg.data_root = args.data_root
    
    if 'val_dataloader' in cfg:
        cfg.val_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.ann_file = args.ann_file
        # Ensure annotations are loaded
        cfg.val_dataloader.dataset.test_mode = False

    # Initialize scope
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    
    # 3. Build Runner and Model
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    runner.model.eval()
    
    print(f"=" * 60)
    print(f"Enhanced Adversarial Point Cloud Visualizer")
    print(f"=" * 60)
    print(f"Model loaded successfully.")
    if args.single:
        print(f"Mode: Single frame (first point cloud only)")
    else:
        print(f"Visualizing every {args.frequency}th frame.")
    print(f"Reading data from: {args.data_root}")
    print(f"Color mode: {args.color_mode}")
    print(f"Theme: {'Dark' if args.dark_theme else 'Light'}")
    print(f"Output directory: {args.output}")
    print(f"=" * 60)
    
    # 4. Iterate
    dataloader = runner.val_dataloader
    processed_count = 0
    
    for idx, data_batch in enumerate(dataloader):
        # Skip frames based on frequency (unless single mode)
        if not args.single and idx % args.frequency != 0:
            continue
            
        data_sample = data_batch['data_samples'][0]
        lidar_filename = os.path.basename(data_sample.lidar_path)
        filename_stem = os.path.splitext(lidar_filename)[0]
        
        print(f"\nProcessing [{idx}]: {filename_stem}")
        
        # Run Inference
        with torch.no_grad():
            outputs = runner.model.test_step(data_batch)
        
        # Extract Data
        points = data_batch['inputs']['points'][0].cpu().numpy()
        gt_boxes_corners = data_sample.gt_instances_3d.bboxes_3d.corners.cpu().numpy()
        gt_labels = data_sample.gt_instances_3d.labels_3d.cpu().numpy()
        
        pred_instances = outputs[0].pred_instances_3d
        scores = pred_instances.scores_3d.cpu().numpy()
        mask = scores > args.show_score_thr
        
        pred_bbox_corners = pred_instances.bboxes_3d[mask].corners.cpu().numpy()
        pred_bbox_labels = pred_instances.labels_3d[mask].cpu().numpy()
        
        display_frame_statistics(
            point_cloud=points,
            gt_boxes_corners=gt_boxes_corners,
            pred_boxes_corners=pred_bbox_corners,
            gt_labels_numeric=gt_labels,
            pred_labels_numeric=pred_bbox_labels,
            output_dir=args.output,
            filename_stem=filename_stem,
            color_mode=args.color_mode,
            dark_theme=args.dark_theme,
            point_size=args.point_size,
            points_keep_ratio=args.points_keep_ratio,
            output_3d=not args.no_3d,
            output_bev=not args.no_bev,
            bev_dpi=args.bev_dpi,
            bev_figsize=tuple(args.bev_figsize)
        )
        
        processed_count += 1
        
        # Exit after first frame if single mode
        if args.single:
            print(f"\n{'=' * 60}")
            print(f"Single frame mode: Exiting after first point cloud.")
            print(f"Images saved to: {args.output}")
            print(f"{'=' * 60}")
            return
    
    print(f"\n{'=' * 60}")
    print(f"Visualization complete!")
    print(f"Processed {processed_count} frames.")
    print(f"Images saved to: {args.output}")
    print(f"{'=' * 60}")




if __name__ == '__main__':
    main()


#usage
'''# Visualize only the first point cloud
python visualize_adv.py \
    --cfg configs/pointpillars_kitti.py \
    --checkpoint model.pth \
    --data_root /path/to/adv_data \
    --ann_file /path/to/kitti_infos_val.pkl \
    --output /path/to/output \
    --single

# Visualize only the first point cloud with density coloring
python visualize_adv.py \
    --cfg configs/pointpillars_kitti.py \
    --checkpoint model.pth \
    --data_root /path/to/adv_data \
    --ann_file /path/to/kitti_infos_val.pkl \
    --output /path/to/output \
    --single \
    --color-mode density

# Full dataset visualization (original behavior)
python visualize_adv.py \
    --cfg configs/pointpillars_kitti.py \
    --checkpoint model.pth \
    --data_root /path/to/adv_data \
    --ann_file /path/to/kitti_infos_val.pkl \
    --output /path/to/output \
    --frequency 20
    '''