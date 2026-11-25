import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

depth_scale = 1.0

parser = ArgumentParser()
parser.add_argument("--cam_path", type=str, default="/path/to/robustmvd/tanks_and_temples")
parser.add_argument("--dataset_path", type=str, default="/path/to/robustmvd/tanks_and_temples")
parser.add_argument("--output_path", type=str, default="/path/to/tanks_and_temples/tnt_nerf_vis")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
scene_lists = ['Courthouse']
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    colmap_path = os.path.join(args.cam_path, scene, "transforms.json")
    with open(colmap_path, 'r') as f:
        cam_params = json.load(f)
    fx, fy, cx, cy = cam_params['fx'], cam_params['fy'], cam_params['cx'], cam_params['cy']
    fx, fy, cx, cy = fx / 2, fy / 2, cx / 2, cy / 2
    fovx = 2 * np.arctan(cx / fx)
    pose_list = cam_params['frames']
    colmap_scale = cam_params['depth_scale']

    trans_info = {'train': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
            'test': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
            'all': { 'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},}


    img_lists = os.listdir(os.path.join(scene_path, "images"))
    img_lists.sort()

    out_rgb_dir = os.path.join(args.output_path, scene, 'input')
    out_depth_dir = os.path.join(args.output_path, scene, 'depth')

    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    if len(img_lists) > 800:
        selected_frame_id = [(i, img) for (i, img) in enumerate(img_lists) if i % 2 == 0]
    else:
        selected_frame_id = [(i, img) for (i, img) in enumerate(img_lists)]

    for img_idx, img in tqdm(selected_frame_id):
        if scene == 'Church' and img_idx >=500:
            break
        img_path = os.path.join(scene_path, "images", img)
        img_name = os.path.basename(img_path).split(".")[0]
        depth_path = os.path.join(scene_path, "depth", f"{(int(img_name) + 1):06d}.npz")
        pose = np.array(pose_list[img_idx]['transform_matrix'], dtype=np.float32)
        pose[:3, 3] *= colmap_scale


        info = {'file_path': f'input/{img_idx:06d}.jpg', 'depth_file_path': f'depth/{img_idx:06d}.exr','transform_matrix': pose.tolist()}

        trans_info['all']['frames'].append(info)
        if (img_idx//2 + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
