import time
import yaml
import pickle
import argparse
import numpy as np
import open3d as o3d
import os


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/visualizer_skeleton_3d.yaml',
    )
    return parser.parse_args()


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def _load_poses(path):
    with open(path, 'rb') as handler:
        poses = pickle.load(handler)
    return np.array(poses, dtype=np.float32)


def _preprocess_poses(raw_poses):
    valid_frames = []
    for frame in raw_poses:
        keypoints = frame.reshape(-1, 3).astype(np.float32)
        keypoints[:, 1] *= -1  # Flip Y axis

        if np.allclose(keypoints, 0):
            continue
        if np.max(np.abs(keypoints)) > 100:
            keypoints /= 1000.0  # Convert mm to meters
        valid_frames.append(keypoints)
    return valid_frames


def _create_geometries(keypoints, connections, point_config, line_config):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    pcd.paint_uniform_color(point_config['color'])

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(keypoints)
    lines.lines = o3d.utility.Vector2iVector(connections)
    lines.colors = o3d.utility.Vector3dVector([line_config['color']] * len(connections))

    return pcd, lines


def visualize_sequence(poses, connections, background_color, point_config, line_config, save_enabled, save_path):
    frame_idx = [0]
    total_frames = len(poses)
    play = [False]

    os.makedirs(save_path, exist_ok=True)

    keypoints = poses[frame_idx[0]]
    pcd, lines = _create_geometries(keypoints, connections, point_config, line_config)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='3D Skeleton Viewer', width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array(background_color, dtype=np.float32)
    render_opt.show_coordinate_frame = True
    render_opt.point_size = point_config.get('size', 5.0)
    render_opt.line_width = line_config.get('width', 1.0)

    vis.add_geometry(pcd)
    vis.add_geometry(lines)

    def update():
        keypoints = poses[frame_idx[0]]
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)
        ctr = vis.get_view_control()
        ctr.set_lookat(np.mean(keypoints, axis=0).tolist())
        vis.update_geometry(pcd)
        vis.update_geometry(lines)
        vis.poll_events()
        vis.update_renderer()

        if save_enabled:
            filename = os.path.join(save_path, f"frame_{frame_idx[0]:04d}.png")
            vis.capture_screen_image(filename, do_render=True)

    def next_frame(vis_):
        if frame_idx[0] < total_frames - 1:
            frame_idx[0] += 1
            update()

    def prev_frame(vis_):
        if frame_idx[0] > 0:
            frame_idx[0] -= 1
            update()

    def toggle_play(vis_):
        play[0] = not play[0]

    def quit_viewer(vis_):
        vis_.close()

    vis.register_key_callback(ord('Q'), quit_viewer)
    vis.register_key_callback(256, quit_viewer)  # ESC
    vis.register_key_callback(262, next_frame)   # Right arrow
    vis.register_key_callback(263, prev_frame)   # Left arrow
    vis.register_key_callback(32, toggle_play)   # Spacebar

    update()

    while vis.poll_events():
        if play[0]:
            if frame_idx[0] < total_frames - 1:
                frame_idx[0] += 1
                update()
            else:
                play[0] = False
        time.sleep(0.03)

    vis.destroy_window()


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    raw_poses = _load_poses(configs['path'])
    halpe_lines = np.array(configs['halpe_lines'], dtype=np.int32)
    background_color = configs.get('background_color', [1.0, 1.0, 1.0])
    point_config = configs.get('point', {'color': [0.0, 1.0, 0.0], 'size': 5.0})
    line_config = configs.get('line', {'color': [0.0, 0.0, 0.0], 'width': 1.0})
    save_enabled = configs.get('save', False)
    save_path = configs.get('save_path', 'artifacts/skeleton_3d')

    poses = _preprocess_poses(raw_poses)
    print(f"🎥 Loaded {len(poses)} valid 3D skeleton frames.")

    if not poses:
        print("❌ No valid frames found.")
    else:
        visualize_sequence(
            poses, halpe_lines,
            background_color,
            point_config, line_config,
            save_enabled, save_path
        )
