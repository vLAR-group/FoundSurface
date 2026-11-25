
import cv2
import numpy as np
import os
from dataset.base.base_stereo_view_dataset import BaseStereoViewDataset
import json
import random
from PIL import Image
from utils.rays import depth_to_ray_distance
import torch
from dataset.utils.data_utils import rotate_90_degrees

class MVS_Synth(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scene_source_dir = []
        scenes_list = os.listdir(self.ROOT)
        scenes_list.sort()
        for scene in scenes_list:
            single_scene_source_dir = os.path.join(self.ROOT, scene)
            assert os.path.exists(single_scene_source_dir)
            self.scene_source_dir.append(single_scene_source_dir)
        if self.split == "train":
            self.transformsfile = "transforms_train.json"
        elif self.split == "test":
            self.transformsfile = "transforms_test.json"
        else:
            self.transformsfile = "transforms_all.json"
        self.intrinsics_list = []
        self.frames_list = []
        self.dpt_scale_list = []
        for scene_source_dir in self.scene_source_dir:
            with open(os.path.join(scene_source_dir, self.transformsfile)) as json_file:
                contents = json.load(json_file)
            self.dpt_scale_list.append(contents["depth_scale"])
            self.frames_list.append(contents["frames"])


    def __len__(self):
        return len(self.scene_source_dir)

    def _get_views(self, idx, resolution, rng):

        rec_data_dict = {"images": [], "depths":[], "Ks": [], "c2ws": []}
        novel_data_dict = {"ray_dists": [], "depths":[], "Ks": [], "c2ws": []}
        path = self.scene_source_dir[idx]
        dpt_scale = self.dpt_scale_list[idx]
        frames = self.frames_list[idx]
        frames_len = len(frames)

        random_num = random.random()
        if random_num < 0.5:
            rec_img_num = 10
            sample_length = min(rec_img_num*2, frames_len - 1)
            start_idx = random.randint(0, frames_len - sample_length - 1)
            end_idx = start_idx + sample_length - 1
            indices = np.linspace(start_idx, end_idx, num=rec_img_num, dtype=int)
            novel_indices = np.random.choice(range(start_idx, end_idx), size=self.novel_view_num)
            novel_frames = [frames[i] for i in novel_indices]
            rec_frames = [frames[i] for i in indices]
        else:
            rec_img_num = random.randint(self.sample_range[0], self.sample_range[1])
            rec_img_num = min(rec_img_num, frames_len)
            indices = np.linspace(0, frames_len - 1, num=rec_img_num, dtype=int)
            novel_indices = np.random.choice(frames_len, size=self.novel_view_num)
            novel_frames = [frames[i] for i in novel_indices]
            rec_frames = [frames[i] for i in indices]

        for viewpoint_idx in range(len(rec_frames)):
            frame = rec_frames[viewpoint_idx]
            fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
            cam_name = os.path.join(path, frame["file_path"])
            dpt_name = os.path.join(path, frame["depth_file_path"])

            c2w = np.array(frame["transform_matrix"])
            if c2w.shape[0] != 4:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            c2w[:3, 1:3] *= -1

            image_path = os.path.join(path, cam_name)
            image = Image.open(image_path)
            width, height = image.size

            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert("RGB")

            depth_path = os.path.join(path, dpt_name)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / dpt_scale

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            new_width = self.vggt_target_size
            new_height = round(height * (new_width / width) / 14) * 14

            image, depth, K = self._crop_resize_if_necessary(image, depth, K, (new_width, new_height), rng=rng)

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = np.array(image)
                image = torch.from_numpy(image).float() / 255.0

            rec_data_dict["images"].append(image)
            rec_data_dict["c2ws"].append(torch.tensor(c2w, dtype=torch.float32))
            rec_data_dict["depths"].append(torch.tensor(depth, dtype=torch.float32))
            rec_data_dict["Ks"].append(torch.tensor(K, dtype=torch.float32))


        for viewpoint_idx in range(len(novel_frames)):
            frame = novel_frames[viewpoint_idx]
            fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
            cam_name = os.path.join(path, frame["file_path"])
            dpt_name = os.path.join(path, frame["depth_file_path"])

            c2w = np.array(frame["transform_matrix"])
            if c2w.shape[0] != 4:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            c2w[:3, 1:3] *= -1


            depth_path = os.path.join(path, dpt_name)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / dpt_scale

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            image = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            if self.random_crop:
                direct_crop = rng.random() > self.crop_prob_threh
                if direct_crop:
                    image, depth, K = self._crop_if_necessary(image, depth, K, resolution, rng=rng)
                else:
                    image, depth, K = self._crop_resize_if_necessary(image, depth, K, resolution, rng=rng)
            else:
                image, depth, K = self._crop_resize_if_necessary(image, depth, K, resolution, rng=rng)

            if self.landscape_check:
                image = np.array(image)
                H, W = image.shape[:2]
                if H > 1.1 * W:
                    clockwise = rng.random() > 0.5
                    image, depth, c2w, K = rotate_90_degrees(image, depth, c2w, K, clockwise=clockwise)
            ray = depth_to_ray_distance(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

            novel_data_dict["c2ws"].append(torch.tensor(c2w, dtype=torch.float32))
            novel_data_dict["ray_dists"].append(torch.tensor(ray, dtype=torch.float32))
            novel_data_dict["depths"].append(torch.tensor(depth, dtype=torch.float32))
            novel_data_dict["Ks"].append(torch.tensor(K, dtype=torch.float32))


        return  rec_data_dict, novel_data_dict
