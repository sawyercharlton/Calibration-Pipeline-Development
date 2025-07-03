"""Extract initial 3D positions of the 2D skeleton keypoints by
multi-camera triangulation
"""

import yaml
import pickle
import argparse
import numpy as np
import pycalib
from tqdm import tqdm


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_triangulation.yaml',
    )
    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def _get_intrinsics(configs):
    with open(configs['intrinsics'], 'rb') as handler:
        intrinsics = pickle.load(handler)

    return intrinsics


def _get_extrinsics(configs):
    with open(configs['extrinsics'], 'rb') as handler:
        extrinsics = pickle.load(handler)

    return extrinsics


def _calc_extrinsics(cam, extrinsics):
    R_global = np.eye(3)
    T_global = np.zeros(3)

    chain = {
        "cam0": ["cam1", "cam2"],
        "cam1": ["cam2"],
        "cam2": [],
    }

    R_chain = []
    T_chain = []

    current = cam
    while chain.get(current):
        next_cam = chain[current][0]
        R = np.array(extrinsics[next_cam]['rotation'], np.float32)
        T = np.array(extrinsics[next_cam]['transition'], np.float32)

        R_chain.append(R)
        T_chain.append(T)

        current = next_cam

    for R, T in reversed(list(zip(R_chain, T_chain))):
        R_global = R @ R_global
        T_global = R @ T_global + T.reshape(3)

    return R_global.tolist(), T_global.tolist()


def calc_3d_skeleton(poses, params, configs):
    cameras = poses.keys()
    length = len(poses[list(cameras)[0]]['pose'])

    num_people = configs['num_people']
    num_points = configs['num_points']
    points_per_timestep = num_people * num_points

    points_3d = []
    for timestep in tqdm(range(length)):
        points_3d_timestep = []
        for point_idx in range(points_per_timestep):
            points_2d = []
            parameters = []
            for camera in cameras:
                points_timestep = np.array(
                    poses[camera]['pose'][timestep]).reshape(-1, 2)
                confidences_timestep = np.array(
                    poses[camera]['pose_confidence'][timestep]).reshape(-1)
                if point_idx < len(points_timestep) and \
                        confidences_timestep[point_idx] > configs['threshold']:
                    points_2d.append(points_timestep[point_idx])
                    
                    cam_mtx = np.array(params[camera]['mtx'], np.float32)
                    extrinsics = np.zeros((3, 4), dtype=float)
                    extrinsics[:3, :3] = np.array(
                        params[camera]['rotation'], np.float32)
                    extrinsics[:3, 3] = np.array(
                        params[camera]['translation'], np.float32)
                    parameters.append(cam_mtx @ extrinsics)

            if len(points_2d) > 0:
                points_2d = np.expand_dims(np.array(points_2d), 1)
                
                parameters = np.array(parameters)
                points_3d_single = pycalib.triangulate_Npts(
                    pt2d_CxPx2=points_2d, P_Cx3x4=parameters)
            else:
                print(f"points_2d is empty in {timestep}")
                points_3d_single = np.zeros((1, 3))
            
            points_3d_timestep.append(points_3d_single)
        points_3d.append(points_3d_timestep)

    points_3d = np.array(points_3d).reshape(length, num_people, num_points, 3)

    return points_3d


def _calc_params(configs):
    intrinsics = _get_intrinsics(configs)
    extrinsics = _get_extrinsics(configs)

    params_global = {}
    for camera in intrinsics.keys():
        rotation, translation = _calc_extrinsics(camera, extrinsics)

        params_global[camera] = {
            'mtx': intrinsics[camera]['mtx'],
            'dist': intrinsics[camera]['dist'],
            'rotation': rotation,
            'translation': translation,
        }

    return params_global


def _filter_by_id(poses, ids):
    poses_filtered = {}
    for camera in poses.keys():
        poses_cam_filtered = []
        poses_conf_filtered = []
        for timestep, pose_timestep in enumerate(poses[camera]['pose']):
            poses_timestep = []
            poses_conf_timestep = []
            for idx_person, person in enumerate(pose_timestep):
                id = poses[camera]['ids'][timestep][idx_person]
                if id in ids:
                    poses_timestep.append(person)
                    poses_conf_timestep.append(
                        poses[camera]['pose_confidence'][timestep][idx_person])

            poses_cam_filtered.append(poses_timestep)
            poses_conf_filtered.append(poses_conf_timestep)
        
        poses_filtered[camera] = {
            'pose': poses_cam_filtered,
            'pose_confidence': poses_conf_filtered,
        }

    return poses_filtered


def _store_artifacts(artifact, output):
    with open(output, 'wb') as handle:
        pickle.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    with open(configs['skeletons'], 'rb') as handler:
        poses = pickle.load(handler)

    # poses = _filter_by_id(poses, [0])
        
    params = _calc_params(configs)
    poses_3d = calc_3d_skeleton(poses, params, configs)

    _store_artifacts(params, configs['output_params'])
    _store_artifacts(poses_3d.tolist(), configs['output'])
