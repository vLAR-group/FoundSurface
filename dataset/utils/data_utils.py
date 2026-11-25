import numpy as np

def rotate_90_degrees(
    image, depth_map, pose_opencv, intri_opencv, clockwise=True
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry. Track coordinates are also updated if provided.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.
        track (np.ndarray or None):
            Optional (N, 2) track array. Will be rotated accordingly.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv,
                new_track
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[:2]

    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)

    new_pose_opencv = adjust_pose_matrix_rot90(pose_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_pose_opencv,
        new_intri_opencv,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise),
    using a transpose+flip pattern.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=1)
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=0)
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
    return np.copy(rotated_image), np.copy(rotated_depth_map)


def adjust_pose_matrix_rot90(pose_c2w, clockwise):
    """
    Rotate a 4x4 camera-to-world pose to match a 90-degree image rotation.

    Args:
        pose_c2w (np.ndarray): 4x4 pose matrix (R | t).
        clockwise (bool): whether the image was rotated clockwise by 90 degrees.

    Returns:
        np.ndarray: the rotated 4x4 pose matrix.
    """
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]

    if clockwise:
        R_rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
    else:
        R_rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

    new_R = np.dot(R, R_rotation)

    new_pose = np.eye(4)
    new_pose[:3, :3] = new_R
    new_pose[:3, 3] = t

    return new_pose


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv
