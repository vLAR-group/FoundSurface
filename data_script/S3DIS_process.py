import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import json
import open3d as o3d

split_dict = {"test":["area_5a", "area_5b"]}
select_interval_eval = 10
s3dis_path = "/path/to/S3DIS/"
error_list = []
for s, area_list in split_dict.items():
    out_dir = f"/path/to/S3DIS/S3DIS_nerf/{s}"
    depth_scale = 512
    for area in area_list:
        area_path = os.path.join(s3dis_path, area, "data")
        depth_path = os.path.join(area_path, "depth")
        scene_dic = {}
        depth_list = [f for f in os.listdir(depth_path) if f.endswith('.png')]
        depth_list.sort()
        try:
            for depth_name in depth_list:
                scene_name = depth_name.split("_")[2] + "_" + depth_name.split("_")[3]
                if scene_name not in scene_dic:
                    scene_dic[scene_name] = []
                scene_dic[scene_name].append(depth_name)

            for scene_name in tqdm(scene_dic.keys()):
                output_scene_name = area+"_"+scene_name
                trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},'test': {"depth_scale": depth_scale, 'frames': []},'all': { "depth_scale": depth_scale, 'frames': []},}
                out_rgb_dir = os.path.join(out_dir, output_scene_name, 'input')
                out_depth_dir = os.path.join(out_dir, output_scene_name, 'depth')
                if "area_5" in area:
                    dense_pts_path = os.path.join(s3dis_path, "processed_pointcloud", "area_5", scene_name+".ply")
                else:
                    dense_pts_path = os.path.join(s3dis_path, "processed_pointcloud", area, scene_name+".ply")
                os.makedirs(out_rgb_dir, exist_ok=True)
                os.makedirs(out_depth_dir, exist_ok=True)
                point_cloud = o3d.io.read_point_cloud(dense_pts_path)
                shutil.copy2(dense_pts_path, os.path.join(out_dir, output_scene_name, output_scene_name+".ply"))
                pcd = point_cloud.voxel_down_sample(voxel_size=0.2)
                o3d.io.write_point_cloud(os.path.join(out_dir, output_scene_name, 'points3d.ply'), pcd)

                for i, depth_name in enumerate(scene_dic[scene_name]):
                    depth_image_path = os.path.join(depth_path, depth_name)
                    rgb_name = depth_name.replace("depth", "rgb")
                    rgb_image_path = os.path.join(area_path, "rgb", rgb_name)
                    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
                    depth[depth>65530] = 0
                    cv2.imwrite(os.path.join(out_depth_dir, depth_name), depth)
                    shutil.copy2(rgb_image_path, os.path.join(out_rgb_dir, rgb_name))
                    pose_file_name = depth_name.replace("depth", "pose").replace("png", "json")
                    pose_file_path = os.path.join(area_path, "pose", pose_file_name)
                    with open(pose_file_path, "r") as f:
                        pose_info = json.load(f)
                        pose = pose_info["camera_rt_matrix"]
                        K = pose_info["camera_k_matrix"]
                        pose = np.array(pose)
                        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
                        pose  = np.linalg.inv(pose)
                        pose[:3, 1:3] *= -1
                    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
                    fovx = 2 * np.arctan(cx / fx)
                    info = {'file_path': f'input/{rgb_name}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{depth_name}',
                            'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx}
                    if (i+1) % select_interval_eval ==0:
                        trans_info['test']['frames'].append(info)
                    else:
                        trans_info['train']['frames'].append(info)
                    trans_info["all"]['frames'].append(info)

                for split in ['train', 'test', 'all']:
                    savedir = os.path.join(out_dir, output_scene_name)
                    with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                        json.dump(trans_info[split], fp, indent=4)
        except Exception as e:
            error_list.append((area, scene_name, e))
print(error_list)
