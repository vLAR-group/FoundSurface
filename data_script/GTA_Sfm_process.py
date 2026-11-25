import h5py
import cv2
import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

depth_scale = 1.0
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/GTA-SfM/train")
parser.add_argument("--output_path", type=str, default="/path/to/GTA-SfM/GTA-SfM_nerf")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    h5file = h5py.File(scene_path, "r")
    img_num = len(h5file.keys()) / 4
    scene_name = os.path.basename(scene_path).split(".")[0]
    out_rgb_dir = os.path.join(args.output_path, scene_name, 'input')
    out_depth_dir = os.path.join(args.output_path, scene_name, 'depth')
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
        'test': { "depth_scale": depth_scale, 'frames': []},
        'all': {"depth_scale": depth_scale, 'frames': []},}

    for frame_idx in range(int(img_num)):
        img_name = f"image_{frame_idx:d}"
        K_name = f"K_{frame_idx:d}"
        pose_name = f"pose_{frame_idx:d}"
        depth_name = f"depth_{frame_idx:d}"

        img = cv2.imdecode(h5file[img_name][:],cv2.IMREAD_COLOR)
        K = h5file[K_name][:]
        pose = h5file[pose_name][:]
        depth = h5file[depth_name][:]


        depth[depth > 300.0] = 0.0


        cv2.imwrite(os.path.join(out_rgb_dir, f"{frame_idx:07d}.jpg"), img)
        cv2.imwrite(os.path.join(out_depth_dir, f"{frame_idx:07d}.exr"), depth)
        cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
        fovx = 2 * np.arctan(cx / fx)
        fovy = 2 * np.arctan(cy / fy)
        pose[:3, 1:3] *= -1
        info = {'file_path': f'input/{frame_idx:07d}.jpg', 'depth_file_path': f'depth/{frame_idx:07d}.exr',
                'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy, "camera_angle_x":fovx, "camera_angle_y":fovy, 'transform_matrix': pose.tolist()}
        trans_info['all']['frames'].append(info)
        if (frame_idx + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene_name)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
