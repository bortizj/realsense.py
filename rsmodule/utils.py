"""
Copyleft 2025
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz-Jaramillo
"""

import json
import gzip
import pickle
import cv2
import csv
import sys
import time
import numpy as np
from typing import Callable
from pathlib import Path

import logging

main_logger = logging.getLogger(__name__)


def setup_logging(data_path: Path | None = None):
    data_path.joinpath("logs").mkdir(parents=True, exist_ok=True)
    log_file_path = data_path.joinpath("logs", "SLAM.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.propagate = False


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator for making the function measure the time and prints the lapsed time in the function
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        main_logger.info(f"Function '{func.__qualname__}' ran in {end_time - start_time:.3f} [sec]!")
        return result

    return wrapper


def compress_dict(data_dict: dict) -> bytes:
    """
    Serializes a dictionary to JSON and then compresses it using gzip
    """
    json_data = json.dumps(data_dict, ensure_ascii=False).encode("utf-8")
    compressed_data = gzip.compress(json_data)

    return compressed_data


def decompress_dict(compressed_data: bytes) -> dict:
    """
    Decompresses gzipped data and then deserializes it from JSON
    """
    decompressed_data = gzip.decompress(compressed_data)
    data_dict = json.loads(decompressed_data.decode("utf-8"))

    return data_dict


def pickle_to_bytes(data_dict: dict) -> bytes:
    """
    Pickles a dictionary into bytes
    """
    pickled_data = pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL)

    return pickled_data


def unpickle_from_bytes(pickled_data):
    """
    Un-pickles data from a bytes object back into a dictionary
    """
    data_dict = pickle.loads(pickled_data)

    return data_dict


def pad_and_hstack_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Pads two images to the same height and horizontally stacks them.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 > h2:
        # Pad img2 to match h1
        pad_amount = h1 - h2
        # (top, bottom, left, right)
        img2_padded = cv2.copyMakeBorder(img2, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img1_padded = img1
    elif h2 > h1:
        # Pad img1 to match h2
        pad_amount = h2 - h1
        img1_padded = cv2.copyMakeBorder(img1, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img2_padded = img2
    else:
        # Heights are already equal, no padding needed
        img1_padded = img1
        img2_padded = img2

    return np.hstack((img1_padded, img2_padded))


def draw_rectangle(
    img: np.ndarray,
    txt: str,
    org_x: int,
    org_y: int,
    font_scale: float = 0.7,
    thickness: int = 2,
    color: tuple = (125, 125, 125),
) -> np.ndarray:
    """
    Draws a filled rectangle around the text on the image
    """
    (text_width, text_height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    padding = 5
    top_left_x = org_x - padding
    top_left_y = org_y - text_height - baseline - padding
    bottom_right_x = org_x + text_width + padding
    bottom_right_y = org_y + baseline + padding

    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1)

    return img


def read_poses_from_file(file_path: Path) -> list:
    """
    Reads poses from a file and returns them as a list of dictionaries.
    Each dictionary contains 'timestamp' and 'pose' keys.
    """
    list_pose_file_ids = range(len(list(file_path.joinpath("camera_poses").glob("id_*.gz"))))
    if file_path.joinpath("list_files.csv").exists():
        with open(file_path.joinpath("list_files.csv"), mode="r", newline="") as file:
            csv_reader = csv.reader(file)
            list_pose_file_ids = next(csv_reader)
            list_pose_file_ids = [int(id) for id in list_pose_file_ids]

    list_poses = []
    for id in list_pose_file_ids:
        pose_file = file_path.joinpath("camera_poses", f"id_{id}.gz")
        with open(pose_file, "rb") as file:
            data = unpickle_from_bytes(file.read())
            list_poses.append(data)

    return list_poses


def compute_reprojection_error(
    points3D_prev: np.ndarray,
    points2D_curr: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    inliers: np.ndarray = None,
):
    image_points_projected, _ = cv2.projectPoints(
        objectPoints=points3D_prev,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )

    # Reshape projected points to be N x 2
    image_points_projected = image_points_projected.reshape(-1, 2)

    # 2. Calculate the reprojection error for all points
    reprojection_errors = np.linalg.norm(points2D_curr - image_points_projected, axis=1)

    # 3. Separate errors for inliers and outliers
    if inliers is not None:
        inlier_indices = inliers.flatten()
        outlier_indices = np.setdiff1d(np.arange(len(points3D_prev)), inlier_indices)

        inlier_errors = reprojection_errors[inlier_indices]
        outlier_errors = reprojection_errors[outlier_indices]
    else:
        outlier_errors = reprojection_errors
        inlier_errors = np.array([])

    return inlier_errors, outlier_errors
