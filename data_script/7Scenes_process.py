import cv2
import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
import pandas as pd
import glob
import re

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

depth_scale = 1000.0
parser = ArgumentParser()
parser.add_argument("--cam_path", type=str, default="/path/to/7Scenes/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
parser.add_argument("--dataset_path", type=str, default="/path/to/7Scenes/raw_data")
parser.add_argument("--output_path", type=str, default="/path/to/7Scenes/7scenes_nerf")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    colmap_path = os.path.join(args.cam_path, f'{scene}_test.txt')

    cam_df = pd.read_csv(colmap_path, sep='\s+', header=None,
                  names=['path', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'f'],
                  index_col=0)

    with open(scene_path + "/TestSplit.txt", "r") as file:
        lines = [int(num) for line in file for num in re.findall(r'\d+', line)]

    for seq_idx_str in lines:
        seq_idx = f"{int(seq_idx_str):02d}"
        seq_name = f"seq-{seq_idx}"
        trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
        'test': { "depth_scale": depth_scale, 'frames': []},
        'all': {"depth_scale": depth_scale, 'frames': []},}

        out_rgb_dir = os.path.join(args.output_path, scene+"_"+seq_name, 'input')
        out_depth_dir = os.path.join(args.output_path, scene+"_"+seq_name, 'depth')

        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)

        img_paths = glob.glob(os.path.join(scene_path, seq_name, "frame-*.color.png"))
        img_paths.sort()
        select_img_paths = [c for idx, c in enumerate(img_paths) if idx % 10 == 0]
        for frame_idx, image_path in enumerate(select_img_paths):
            image_name = os.path.basename(image_path).replace(".color.png", ".png")
            depth_path = image_path.replace("color", "depth")


            pose_index = image_path.split("/")[-2] + "/" + image_path.split("/")[-1]
            if pose_index not in cam_df.index:
                print(f"Pose index {pose_index} not found in camera data for scene {scene}, skipping.")
                continue
            qw, qx, qy, qz, tx, ty, tz, f = cam_df.loc[pose_index]
            cx, cy = 320.0, 240.0
            R_w2c = qvec2rotmat([qw, qx, qy, qz])
            t_w2c = np.array([tx, ty, tz])
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            pose = np.eye(4)
            pose[:3, :3] = R_c2w
            pose[:3, 3] = t_c2w
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth[depth>60000] = 0
            cv2.imwrite(os.path.join(out_depth_dir, image_name), depth)
            fovx = 2 * np.arctan(cx / f)


            pose[:3, 1:3] *= -1
            info = {'file_path': f'input/{image_name}', 'depth_file_path': f'depth/{image_name}', 'cx': cx, 'cy': cy, 'fx': f,
                    'fy':f, "camera_angle_x":fovx, 'transform_matrix': pose.tolist()}

            trans_info['all']['frames'].append(info)
            if (frame_idx + 1) % 10 == 0:
                trans_info['test']['frames'].append(info)
            else:
                trans_info['train']['frames'].append(info)

        for split in ['train', 'test', 'all']:
            savedir = os.path.join(args.output_path, scene+"_"+seq_name)
            with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                json.dump(trans_info[split], fp, indent=4)
