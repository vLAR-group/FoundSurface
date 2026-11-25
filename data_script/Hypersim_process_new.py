import h5py
import numpy as np
import os
from PIL import Image
import json
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from utils import convert_d


def opengl_to_intrinsics(proj_matrix, width_pixels, height_pixels):
    K00 = proj_matrix[0, 0] * width_pixels / 2.0
    K01 = -proj_matrix[0, 1] * width_pixels / 2.0
    K02 = (1.0 - proj_matrix[0, 2]) * width_pixels / 2.0
    K11 = proj_matrix[1, 1] * height_pixels / 2.0
    K12 = (1.0 + proj_matrix[1, 2]) * height_pixels / 2.0
    return np.array([[K00, K01, K02], [0.0, K11, K12], [0.0, 0.0, 1.0]])

depth_scale = 1000.0

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/ml-hypersim/evermotion_dataset/scenes")
parser.add_argument("--output_path", type=str, default="/path/to/Hypersim/Hypersim_nerf_new")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
scene_lists = scene_lists[45:]
metadata_camera_parameters_csv_file = os.path.join("/path/to/ml-hypersim/contrib/mikeroberts3000", "metadata_camera_parameters.csv")
df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

gamma = 1.0 / 2.2
inv_gamma = 1.0 / gamma
percentile = 90
brightness_nth_percentile_desired = 0.8

for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    if not os.path.isdir(os.path.join(scene_path, "_detail")):
        print(f"Scene path {scene_path} is not a directory, skipping.")
        continue
    if not os.path.exists(os.path.join(scene_path, "_detail", "metadata_cameras.csv")):
        print(f"Metadata cameras CSV not found for scene {scene}, skipping.")
        continue
    camera_ids = (
        pd.read_csv(
            os.path.join(scene_path, "_detail", "metadata_cameras.csv"),
            header=None,
            skiprows=1,
        )
        .to_numpy()
        .flatten()
    )
    df_ = df_camera_parameters.loc[scene]
    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])
    meters_per_asset_unit = df_["settings_units_info_meters_scale"]

    M_proj = np.array(
        [
            [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
            [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
            [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
            [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
        ]
    )
    K = opengl_to_intrinsics(
        M_proj, width_pixels, height_pixels
    ).astype(np.float32)
    if K[0, 1] != 0:
        print(f"camera_intrinsics[0, 1] != 0: {K[0, 1]}")
        continue
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    fovx = 2 * np.arctan(cx / fx)

    for camera_name in camera_ids:
        camera_dir = os.path.join(scene_path, "_detail", camera_name)
        if not os.path.exists(camera_dir):
            print(f"Camera directory not found for scene {scene}, skipping.")
            continue
        with h5py.File(os.path.join(camera_dir, "camera_keyframe_frame_indices.hdf5"), "r") as f:
            keyframe_indices = f["dataset"][:]

        img_dir = os.path.join(scene_path, "images", f"scene_{camera_name}_final_hdf5")
        depth_dir = os.path.join(scene_path, "images", f"scene_{camera_name}_geometry_hdf5")
        out_rgb_dir = os.path.join(args.output_path, scene+f"_{camera_name}", 'input')
        out_depth_dir = os.path.join(args.output_path, scene+f"_{camera_name}", 'depth')
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)

        trans_info = {'train': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
                'test': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
                'all': { 'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},}


        camera_positions_hdf5_file    = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
        camera_orientations_hdf5_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")

        with h5py.File(camera_positions_hdf5_file, "r") as f:
            camera_positions = f["dataset"][:]
        with h5py.File(camera_orientations_hdf5_file, "r") as f:
            camera_orientations = f["dataset"][:]

        assert len(camera_positions) == len(
            camera_orientations
        ), f"len(camera_positions)={len(camera_positions)} != len(camera_orientations)={len(camera_orientations)}"

        for frame_idx in keyframe_indices:
            frame_id = str(frame_idx).zfill(4)
            rgb_path = os.path.join(img_dir, f"frame.{frame_id}.color.hdf5")
            depth_path = os.path.join(depth_dir, f"frame.{frame_id}.depth_meters.hdf5")
            if not os.path.exists(rgb_path):
                continue
            with h5py.File(rgb_path, "r") as f:
                img = f["dataset"][:]
            with h5py.File(depth_path, "r") as f:
                distance = f["dataset"][:].astype(np.float32)
            render_entity = depth_path.replace("depth_meters.hdf5", "render_entity_id.hdf5")
            with h5py.File(render_entity, "r") as f:
                render_entity_id = f["dataset"][:].astype(np.int32)
            assert (render_entity_id != 0).all()
            valid_mask = render_entity_id != -1
            if np.sum(valid_mask) == 0:
                scale = 1.0
            else:
                brightness = (
                    0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]
                )
                brightness_valid = brightness[valid_mask]
                eps = 0.0001
                brightness_nth_percentile_current = np.percentile(
                    brightness_valid, percentile
                )
                if brightness_nth_percentile_current < eps:
                    scale = 0.0
                else:
                    scale = (
                        np.power(brightness_nth_percentile_desired, inv_gamma)
                        / brightness_nth_percentile_current
                    )

            img = np.power(np.maximum(scale * img, 0), gamma)
            img = np.clip(img, 0.0, 1.0)

            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(out_rgb_dir, f"{frame_id}.jpg"))
            depth = convert_d(distance, fx, out='dep')
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[depth > 20.0] = 0.0

            camera_position_world = camera_positions[frame_idx]
            R_world_from_cam = camera_orientations[frame_idx]
            t_world_from_cam = np.array(camera_position_world).reshape(3, 1)
            t_world_from_cam = t_world_from_cam * meters_per_asset_unit
            T_world_from_cam = np.concatenate([R_world_from_cam, t_world_from_cam], axis=1)
            pose = np.concatenate([T_world_from_cam, np.array([[0, 0, 0, 1]])], axis=0)


            if not np.isfinite(pose).all():
                print(f"frame_id={frame_id} pose is not finite.")
                continue

            depth = (depth * 1000).astype(np.uint16)
            Image.fromarray(depth).save(os.path.join(out_depth_dir, f"{frame_id}.png"))

            info = {'file_path': f'input/{frame_id}.jpg', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{frame_id}.png'}
            trans_info['all']['frames'].append(info)
            if (frame_idx + 1) % 10 == 0:
                trans_info['test']['frames'].append(info)
            else:
                trans_info['train']['frames'].append(info)

        for split in ['train', 'test', 'all']:
            savedir = os.path.join(args.output_path, scene+f"_{camera_name}")
            with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                json.dump(trans_info[split], fp, indent=4)
