import numpy as np
import plotly.graph_objects as go
from mmdet3d.datasets import build_dataset
from mmcv import Config, load
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize nuScenes 3D Detections')
    parser.add_argument('config', help='config file path used to build the dataset')
    parser.add_argument('predictions', help='prediction .pkl file produced by tools/test.py --out')
    parser.add_argument('--dataset-split', default='val', choices=['train', 'val', 'test'], help='dataset split to visualize')
    parser.add_argument('--sample-idx', type=int, default=0, help='index of sample in the selected dataset split')
    parser.add_argument('--sample-stride', type=int, default=None, help='render every Nth sample instead of one sample')
    parser.add_argument('--multi-scene', action='store_true', help='render multiple samples into one navigable HTML')
    parser.add_argument('--max-scenes', type=int, default=20, help='maximum scenes to embed in --multi-scene mode; set <= 0 for no cap')
    parser.add_argument('--out-html', default='projects/adv_aug/runs/adhoc/visualizations/visualization.html', help='output html path')
    parser.add_argument('--out-dir', default='projects/adv_aug/runs/adhoc/visualizations', help='output directory for non-combined stride mode')
    parser.add_argument('--score-thr', type=float, default=0.35, help='score threshold for predictions')
    parser.add_argument('--point-downsample', type=int, default=10, help='plot every Nth point for browser performance')
    return parser.parse_args()

def draw_box_plotly(fig, corners, color, name, legend_group, show_legend=True, hover_info=None, visible=True):
    # corners is shape (8, 3)
    # The 12 edges are drawn by connecting the corners in a specific order
    # Corners mapping:
    # 0: [-w/2, -l/2, -h/2], 1: [w/2, -l/2, -h/2], 2: [w/2, l/2, -h/2], 3: [-w/2, l/2, -h/2]
    # 4: [-w/2, -l/2, h/2], 5: [w/2, -l/2, h/2], 6: [w/2, l/2, h/2], 7: [-w/2, l/2, h/2]

    # We can connect them using line segments. None is used to lift the pen.
    seq = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, None, 1, 5, None, 2, 6, None, 3, 7]

    x = [corners[idx][0] if idx is not None else None for idx in seq]
    y = [corners[idx][1] if idx is not None else None for idx in seq]
    z = [corners[idx][2] if idx is not None else None for idx in seq]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=color, width=4),
        name=name,
        legendgroup=legend_group,
        showlegend=show_legend,
        text=hover_info,
        hoverinfo='text' if hover_info is not None else 'name',
        visible=visible
    ))

def get_dataset_cfg(cfg, split):
    dataset_cfg = cfg.data[split]
    if split == 'test' and isinstance(dataset_cfg, dict):
        dataset_cfg.test_mode = True
    return dataset_cfg

def add_sample_traces(fig, dataset, predictions, sample_idx, score_thr, point_downsample, visible=True):
    if sample_idx < 0:
        raise IndexError(f'sample index {sample_idx} must be non-negative')
    if sample_idx >= len(dataset):
        raise IndexError(f'sample index {sample_idx} is outside dataset length {len(dataset)}')
    if sample_idx >= len(predictions):
        raise IndexError(f'sample index {sample_idx} is outside prediction length {len(predictions)}')

    # Get info
    info = dataset.data_infos[sample_idx]
    lidar_path = os.path.join(dataset.data_root, info['lidar_path'])
    print(f"Sample index: {sample_idx}")
    print(f"Lidar Path: {lidar_path}")

    # Load points from file (raw points for plotting)
    raw_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points_xyz = raw_points[:, :3]

    # Downsample points for plotly performance
    points_xyz_down = points_xyz[::point_downsample]
    print(f"Original point cloud size: {len(points_xyz)}, downsampled to: {len(points_xyz_down)}")

    # Load ground truth annotations
    ann_info = dataset.get_ann_info(sample_idx)
    gt_bboxes = ann_info['gt_bboxes_3d']
    gt_labels = ann_info['gt_labels_3d']

    # Get stored predictions
    prediction = predictions[sample_idx]
    pred_bbox_results = prediction['pts_bbox'] if 'pts_bbox' in prediction else prediction
    pred_boxes = pred_bbox_results['boxes_3d']
    pred_scores_tensor = pred_bbox_results['scores_3d']
    pred_labels_tensor = pred_bbox_results['labels_3d']

    # Add Point Cloud
    fig.add_trace(go.Scatter3d(
        x=points_xyz_down[:, 0],
        y=points_xyz_down[:, 1],
        z=points_xyz_down[:, 2],
        mode='markers',
        marker=dict(
            size=1.2,
            color=points_xyz_down[:, 2], # Color by height (z-coordinate)
            colorscale='Viridis',
            opacity=0.6
        ),
        name=f'Point Cloud ({sample_idx})',
        legendgroup=f'Point Cloud {sample_idx}',
        visible=visible
    ))

    # Plot Ground Truth Boxes
    gt_corners = gt_bboxes.corners.cpu().numpy()
    class_names = dataset.CLASSES

    print(f"Number of Ground Truth boxes: {len(gt_corners)}")
    for i, corners in enumerate(gt_corners):
        label_idx = gt_labels[i]
        class_name = class_names[label_idx] if label_idx < len(class_names) else 'unknown'
        hover_text = f"GT: {class_name}"
        draw_box_plotly(
            fig, corners,
            color='rgba(46, 204, 113, 0.9)', # Emerald green
            name=f"GT: {class_name}",
            legend_group=f'Ground Truth {sample_idx}',
            show_legend=(i == 0),
            hover_info=hover_text,
            visible=visible
        )

    # Plot Predicted Boxes
    keep_mask = pred_scores_tensor >= score_thr
    filtered_pred_boxes = pred_boxes[keep_mask]
    filtered_scores = pred_scores_tensor[keep_mask].cpu().numpy()
    filtered_labels = pred_labels_tensor[keep_mask].cpu().numpy()

    pred_corners = filtered_pred_boxes.corners.cpu().numpy()
    print(f"Number of predicted boxes (score >= {score_thr}): {len(pred_corners)}")

    for i, corners in enumerate(pred_corners):
        label_idx = filtered_labels[i]
        score = filtered_scores[i]
        class_name = class_names[label_idx] if label_idx < len(class_names) else 'unknown'
        hover_text = f"Pred: {class_name}<br>Score: {score:.2f}"
        draw_box_plotly(
            fig, corners,
            color='rgba(231, 76, 60, 0.9)', # Alizarin red
            name=f"Pred: {class_name} ({score:.2f})",
            legend_group=f'Predictions {sample_idx}',
            show_legend=(i == 0),
            hover_info=hover_text,
            visible=visible
        )

    return {
        'sample_idx': sample_idx,
        'title': f"3D Object Detection Visualization - Sample {sample_idx}",
        'lidar_path': lidar_path,
        'num_points': len(points_xyz),
        'num_points_downsampled': len(points_xyz_down),
        'num_gt': len(gt_corners),
        'num_pred': len(pred_corners),
    }


def configure_layout(fig, title):
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (Right)', backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="white"),
            yaxis=dict(title='Y (Forward)', backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="white"),
            zaxis=dict(title='Z (Up)', backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="white", range=[-10, 10]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='black',
        font=dict(color='white')
    )


def render_sample(dataset, predictions, sample_idx, out_html, score_thr, point_downsample):
    fig = go.Figure()
    scene_info = add_sample_traces(
        fig=fig,
        dataset=dataset,
        predictions=predictions,
        sample_idx=sample_idx,
        score_thr=score_thr,
        point_downsample=point_downsample)
    configure_layout(fig, scene_info['title'])

    # Create docs folder if doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)

    # Save as HTML
    print(f"Saving interactive visualization to {out_html}...")
    fig.write_html(out_html)


def render_multi_scene(dataset, predictions, sample_indices, out_html, score_thr, point_downsample):
    if not sample_indices:
        raise ValueError('no samples selected for --multi-scene')

    fig = go.Figure()
    scenes = []
    trace_ranges = []

    for scene_num, sample_idx in enumerate(sample_indices):
        start_trace = len(fig.data)
        scene_info = add_sample_traces(
            fig=fig,
            dataset=dataset,
            predictions=predictions,
            sample_idx=sample_idx,
            score_thr=score_thr,
            point_downsample=point_downsample,
            visible=(scene_num == 0))
        end_trace = len(fig.data)
        scenes.append(scene_info)
        trace_ranges.append((start_trace, end_trace))

    total_traces = len(fig.data)
    steps = []
    buttons = []

    for scene_num, scene_info in enumerate(scenes):
        visible = [False] * total_traces
        start_trace, end_trace = trace_ranges[scene_num]
        for trace_idx in range(start_trace, end_trace):
            visible[trace_idx] = True

        label = f"{scene_info['sample_idx']}"
        title = scene_info['title']
        args = [{'visible': visible}, {'title': title}]
        steps.append(dict(
            method='update',
            label=label,
            args=args
        ))
        buttons.append(dict(
            method='update',
            label=f"Sample {label}",
            args=args
        ))

    configure_layout(fig, scenes[0]['title'])
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Sample index: '),
            pad=dict(t=35),
            steps=steps
        )],
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            x=0.0,
            xanchor='left',
            y=1.08,
            yanchor='top'
        )],
        margin=dict(l=0, r=0, b=0, t=80)
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    print(f"Saving navigable multi-scene visualization to {out_html}...")
    fig.write_html(out_html)

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    print(f"Loading {args.dataset_split} dataset...")
    dataset = build_dataset(get_dataset_cfg(cfg, args.dataset_split))
    print(f"Dataset samples: {len(dataset)}")

    print(f"Loading predictions from {args.predictions}...")
    predictions = load(args.predictions)
    print(f"Prediction samples: {len(predictions)}")

    if args.multi_scene:
        sample_stride = args.sample_stride or 1
        if sample_stride <= 0:
            raise ValueError('--sample-stride must be a positive integer')

        max_samples = min(len(dataset), len(predictions))
        sample_indices = list(range(args.sample_idx, max_samples, sample_stride))
        if args.max_scenes > 0:
            sample_indices = sample_indices[:args.max_scenes]

        print(f"Rendering {len(sample_indices)} samples into {args.out_html}")
        render_multi_scene(
            dataset=dataset,
            predictions=predictions,
            sample_indices=sample_indices,
            out_html=args.out_html,
            score_thr=args.score_thr,
            point_downsample=args.point_downsample)
        print("Done!")
        return

    if args.sample_stride is None:
        render_sample(
            dataset=dataset,
            predictions=predictions,
            sample_idx=args.sample_idx,
            out_html=args.out_html,
            score_thr=args.score_thr,
            point_downsample=args.point_downsample)
        print("Done!")
        return

    if args.sample_stride <= 0:
        raise ValueError('--sample-stride must be a positive integer')

    max_samples = min(len(dataset), len(predictions))
    sample_indices = list(range(0, max_samples, args.sample_stride))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Rendering {len(sample_indices)} samples to {args.out_dir}")

    for sample_idx in sample_indices:
        out_html = os.path.join(args.out_dir, f'sample_{sample_idx:06d}.html')
        render_sample(
            dataset=dataset,
            predictions=predictions,
            sample_idx=sample_idx,
            out_html=out_html,
            score_thr=args.score_thr,
            point_downsample=args.point_downsample)

    print("Done!")

if __name__ == '__main__':
    main()
