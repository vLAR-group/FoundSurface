import numpy as np
import os
from PIL import Image
import json
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
R_conv = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)
depth_scale = 10.0

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/MVS-Synth/GTAV_540")
parser.add_argument("--output_path", type=str, default="/path/to/MVS_Synth/MVS_Synth_nerf")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    img_dir = os.path.join(scene_path, "images")
    depth_dir = os.path.join(scene_path, "depths")
    pose_dir = os.path.join(scene_path, "poses")
    out_rgb_dir = os.path.join(args.output_path, scene, 'input')
    out_depth_dir = os.path.join(args.output_path, scene, 'depth')
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
        'test': { "depth_scale": depth_scale, 'frames': []},
        'all': {"depth_scale": depth_scale, 'frames': []},}

    rgb_images = os.listdir(img_dir)
    rgb_images.sort()

    for frame_idx, img_name in enumerate(rgb_images):

        rgb_img_path = os.path.join(img_dir, img_name)
        depth_path = rgb_img_path.replace("images", "depths").replace(".png", ".exr")
        pose_path = rgb_img_path.replace("images", "poses").replace(".png", ".json")

        img = Image.open(rgb_img_path)
        img.save(os.path.join(out_rgb_dir, img_name))

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = depth * depth_scale
        cv2.imwrite(os.path.join(out_depth_dir, img_name.replace(".png", ".exr")), depth)


        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        extrinsic = np.array(pose_data["extrinsic"], dtype=np.float32)
        cx,cy,fx,fy = pose_data["c_x"], pose_data["c_y"], pose_data["f_x"], pose_data["f_y"]
        pose = np.linalg.inv(extrinsic)
        pose = R_conv @ pose


        pose[:3, 1:3] *= -1
        fovx = 2 * np.arctan(cx / fx)
        info = {'file_path': f'input/{img_name}', 'depth_file_path': f'depth/{img_name.replace(".png", ".exr")}', 'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy, "camera_angle_x":fovx, 'transform_matrix': pose.tolist()}
        trans_info['all']['frames'].append(info)
        if (frame_idx + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
