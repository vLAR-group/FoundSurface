import numpy as np
import open3d as o3d
import warnings
import re
from scipy.spatial.transform import Rotation
import PIL.Image
import os
import PIL
from tqdm import tqdm
import cv2
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC
import os.path as osp

def convert_dpt2pt(depth, K, pose):
    H, W = depth.shape[0], depth.shape[1]
    x, y = np.meshgrid(np.arange(W, dtype=np.float32),
                            np.arange(H, dtype=np.float32))
    cam_pt = np.stack([(x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], K[2, 2] * np.ones_like(x) * depth, np.ones_like(x)], -1)
    pt = (pose[None, None, :, :]@cam_pt[:, :, :, None]).squeeze()[:, :, :3]
    return pt

def save_pts(pts, path, m=None):
    if m is not None:
        pts = pts[m[...,0]==1]
    pts = np.reshape(pts, [-1, 3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if pts.shape[0] > 5000000:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        warnings.warn("Warning: Point cloud has more than 5 million points, downsampling to reduce size.")
    o3d.io.write_point_cloud(path, pcd)

def convert_d(d, focal, out='dist'):
    H, W = d.shape[0], d.shape[1]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    L = np.sqrt(np.power(j - H / 2., 2) + np.power(i - W / 2., 2) + focal ** 2)
    fl = focal / L
    if out == 'dist':
        return d / fl
    elif out == 'dep':
        return d * fl
    else:
        raise NotImplementedError

REGEXPR_DSLR = re.compile(r'^DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'frame_(?P<frameid>\d+).jpg$')

def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    matches = re.match(regex_expr, name)
    return matches['frameid']


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)

def load_sfm(sfm_dir, cam_type='dslr'):
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    return img_idx, img_infos

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]

def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap)
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix

def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics
