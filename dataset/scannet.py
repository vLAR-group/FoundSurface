
import numpy as np
import os
from dataset.base.base_stereo_view_dataset import BaseStereoViewDataset
from dataset.base.base_eval_dataset import BaseEvalDataset
import json
import random
from PIL import Image
from utils.rays import depth_to_ray_distance
import torch
from dataset.utils.data_utils import rotate_90_degrees

class ScanNet(BaseStereoViewDataset):
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
            fx, fy, cx, cy = contents['fx'], contents['fy'], contents['cx'], contents['cy']
            self.intrinsics_list.append({"fx": fx, "fy": fy, "cx": cx, "cy": cy})

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
            rec_img_num = random.randint(5, 10)
            sample_length = min(random.randint(rec_img_num*3, rec_img_num*10), frames_len - 1)
            start_idx = random.randint(0, frames_len - sample_length - 1)
            end_idx = start_idx + sample_length - 1
            indices = np.linspace(start_idx, end_idx, num=rec_img_num, dtype=int)
            novel_indices = np.random.choice(range(start_idx, end_idx), size=self.novel_view_num, replace=False)
            novel_frames = [frames[i] for i in novel_indices]
            rec_frames = [frames[i] for i in indices]
        elif 0.5 < random_num < 0.9:
            rec_img_num = random.randint(self.sample_range_short[0], self.sample_range_short[1])
            start_idx = random.randint(0, frames_len // 2 - 1)
            indices = np.linspace(start_idx, start_idx + frames_len // 2, num=rec_img_num, dtype=int)
            novel_indices = np.random.choice(range(start_idx, start_idx + frames_len // 2), size=self.novel_view_num, replace=False)
            novel_frames = [frames[i] for i in novel_indices]
            rec_frames = [frames[i] for i in indices]
        else:
            rec_img_num = random.randint(self.sample_range[0], self.sample_range[1])
            rec_img_num = min(rec_img_num, frames_len)
            indices = np.linspace(0, frames_len - 1, num=rec_img_num, dtype=int)
            novel_indices = np.random.choice(frames_len, size=self.novel_view_num, replace=False)
            novel_frames = [frames[i] for i in novel_indices]
            rec_frames = [frames[i] for i in indices]

        fx, fy, cx, cy = self.intrinsics_list[idx]['fx'], self.intrinsics_list[idx]['fy'], self.intrinsics_list[idx]['cx'], self.intrinsics_list[idx]['cy']
        for viewpoint_idx in range(len(rec_frames)):
            frame = rec_frames[viewpoint_idx]
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
            depth = np.array(Image.open(depth_path)) / dpt_scale

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
            cam_name = os.path.join(path, frame["file_path"])
            dpt_name = os.path.join(path, frame["depth_file_path"])

            c2w = np.array(frame["transform_matrix"])
            if c2w.shape[0] != 4:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            c2w[:3, 1:3] *= -1

            depth_path = os.path.join(path, dpt_name)
            depth = np.array(Image.open(depth_path)) / dpt_scale

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            image = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
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

class ScannetEval(BaseEvalDataset):
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

        self.dpt_scale_list, self.intrinsics_list = [], []
        self.rec_frames, self.novel_frames = [], []
        for scene_source_dir in self.scene_source_dir:
            with open(os.path.join(scene_source_dir, "transforms_train.json")) as json_file:
                contents = json.load(json_file)
            self.rec_frames.append(contents["frames"])
            self.dpt_scale_list.append(contents["depth_scale"])
            fx, fy, cx, cy = contents['fx'], contents['fy'], contents['cx'], contents['cy']
            self.intrinsics_list.append({"fx": fx, "fy": fy, "cx": cx, "cy": cy})

            with open(os.path.join(scene_source_dir, f"transforms_{self.split}.json")) as json_file:
                contents = json.load(json_file)
            self.novel_frames.append(contents["frames"])

    def __len__(self):
        return len(self.scene_source_dir)

    def _get_views(self, idx, resolution, rng):

        rec_data_dict = {"images": [], "depths":[], "Ks": [], "c2ws": []}
        novel_data_dict = {"images": [], "ray_dists": [], "depths":[], "Ks": [], "c2ws": [], "org_Ks": []}
        scene_info = {"image_names": [], "scene_name": "", "dataset_name": "", "rec_names": []}
        path = self.scene_source_dir[idx]
        scene_name, dataset_name = path.split("/")[-1], path.split("/")[-3]
        scene_info["scene_name"] = scene_name
        scene_info["dataset_name"] = dataset_name
        dpt_scale = self.dpt_scale_list[idx]
        rec_frames = self.rec_frames[idx]
        indices = np.linspace(0, len(rec_frames)-1, num=self.rec_views, dtype=int)
        rec_frames = [rec_frames[i] for i in indices]
        novel_frames = self.novel_frames[idx]

        fx, fy, cx, cy = self.intrinsics_list[idx]['fx'], self.intrinsics_list[idx]['fy'], self.intrinsics_list[idx]['cx'], self.intrinsics_list[idx]['cy']
        for viewpoint_idx in range(len(rec_frames)):
            frame = rec_frames[viewpoint_idx]
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
            depth = np.array(Image.open(depth_path)) / dpt_scale

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
            scene_info["rec_names"].append(frame["file_path"].split("/")[-1])

        for viewpoint_idx in range(len(novel_frames)):
            frame = novel_frames[viewpoint_idx]
            cam_name = os.path.join(path, frame["file_path"])
            dpt_name = os.path.join(path, frame["depth_file_path"])

            c2w = np.array(frame["transform_matrix"])
            if c2w.shape[0] != 4:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            c2w[:3, 1:3] *= -1

            image_path = os.path.join(path, cam_name)
            image = Image.open(image_path)
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert("RGB")

            depth_path = os.path.join(path, dpt_name)
            depth = np.array(Image.open(depth_path)) / dpt_scale

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            org_K = K.copy()
            ray = depth_to_ray_distance(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            if self.fix_res:
                image, depth, K = self._crop_resize_if_necessary(image, depth, K, (new_width, new_height), rng=rng)
            else:
                image, depth, K = self._crop_resize_if_necessary(image, depth, K, resolution, rng=rng)

            novel_data_dict["images"].append(torch.tensor(np.array(image)))
            novel_data_dict["c2ws"].append(torch.tensor(c2w, dtype=torch.float32))
            novel_data_dict["ray_dists"].append(torch.tensor(ray, dtype=torch.float32))
            novel_data_dict["depths"].append(torch.tensor(depth, dtype=torch.float32))
            novel_data_dict["Ks"].append(torch.tensor(K, dtype=torch.float32))
            novel_data_dict["org_Ks"].append(torch.tensor(org_K, dtype=torch.float32))
            scene_info["image_names"].append(frame["file_path"].split("/")[-1])

        return  rec_data_dict, novel_data_dict, scene_info
