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

import cv2
import numpy as np
import open3d as o3d
import threading

import copy

from rsmodule.utils import timing_decorator, compute_reprojection_error


class VisualSLAM:
    """
    Object to perform Visual SLAM using Intel RealSense camera data.
    """

    def __init__(
        self,
        cam_int: tuple,
        dist_coeffs: np.ndarray = np.zeros((5,)),
        nmatches: int = 100,
        current_voxel_size: float = 0.005,
    ):
        # Camera Intrinsics for calibration
        self.dist_coeffs = dist_coeffs
        self.fx, self.fy, self.cx, self.cy, self.width, self.height = cam_int
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

        # Camera matrix (intrinsics) for OpenCV PnP and the distortion coefficients
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = self.dist_coeffs.reshape((1, 5))

        # Initialize the parameters for the SLAM system
        self.nmatches = nmatches
        self.current_voxel_size = current_voxel_size

        # Current map point cloud
        self.current_map_pcd = o3d.geometry.PointCloud()
        self.current_map_pcd.points = o3d.utility.Vector3dVector()
        self.current_map_pcd.colors = o3d.utility.Vector3dVector()
        self.current_map_pcd.normals = o3d.utility.Vector3dVector()
        self.current_camera_pose = np.eye(4)

        # For camera trajectory visualization
        self.camera_trajectory_points = []

        # Feature detector ORB and brute force matcher for the visual odometry
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Last frame data for pose estimation
        self.process_thread = None
        self.last_frame_data = None
        self.is_processing = False
        self.merge_count = 0

        # Variable to protected the shared data access
        self.data_lock = threading.Lock()

        print("[INFO]: Visual SLAM system initialized.")

    def _project_2d_into_3d(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        Projects a 2D pixel (u, v) with a given depth to its 3D coordinate.
        """
        if depth == 0:
            return np.array([0, 0, 0])

        # Given in the units of the camera's depth scale (usually meters)
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])

    @timing_decorator
    def _estimate_pose_from_features(self, prev_frame_data: dict, curr_frame_data: dict) -> np.ndarray:
        """
        Estimates the relative pose (T_curr_prev) from previous to current frame using feature matching and RANSAC PnP.
        """
        prev_bgr = prev_frame_data["bgr_image"]
        curr_bgr = curr_frame_data["bgr_image"]
        prev_depth = prev_frame_data["depth_image"]

        # Convert to grayscale for feature detection
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # Detect ORB key-points and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print("[Error]: Not enough key-points for pose estimation.")
            return None

        # Match descriptors and Sort them in the order of their distance
        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # For simplicity here, let's take a good number of top matches
        num_matches_to_use = min(len(matches), self.nmatches)
        good_matches = matches[:num_matches_to_use]

        # PnP (Perspective-n-Point) needs at least 4 points
        if len(good_matches) < 4:
            print("[Error]: Not enough good matches for pose estimation.")
            return None

        # Getting the quality of the matches
        distances = [m.distance for m in good_matches]

        # TODO for now just printing as log the values, later we can use this to filter outliers
        print(f"[INFO]: Number of good matches: {len(good_matches)}")
        print(f"[INFO]: Average distance of good matches: {np.mean(distances):.6f}")

        # PnP to solve for pose.
        # 3D points (from world or previous frame) and their corresponding 2D projections (in current image).
        points3D_prev = []
        points2D_curr = []

        for m in good_matches:
            # Get 2D keypoint coordinates from previous frame
            u1, v1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
            # Get 2D keypoint coordinates from current frame
            u2, v2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])

            # Get depth from previous frame for the 3D point
            depth_at_prev_kp = prev_depth[v1, u1]

            # Project 2D point (u1, v1) from previous frame to 3D using its depth
            p3d = self._project_2d_into_3d(u1, v1, depth_at_prev_kp)

            # Only use valid 3D points (i.e., where depth is not 0 or invalid)
            if p3d[2] > 0.01 and p3d[2] < 5.0:
                points3D_prev.append(p3d)
                points2D_curr.append((u2, v2))

        if len(points3D_prev) < 4:
            print("[Error]: Not enough valid 3D-2D correspondences for PnP.")
            return None

        points3D_prev = np.array(points3D_prev, dtype=np.float32)
        points2D_curr = np.array(points2D_curr, dtype=np.float32)

        # Solve PnP using RANSAC for robust estimation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3D_prev,
            imagePoints=points2D_curr,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            reprojectionError=3.0,
            iterationsCount=200,
        )

        if not success:
            print("[Error]: PnP failed to estimate pose.")
            return None

        # Computing the reprojection error
        inlier_errors, outlier_errors = compute_reprojection_error(
            points3D_prev, points2D_curr, rvec, tvec, self.camera_matrix, self.dist_coeffs, inliers
        )

        # TODO for now just printing as log the values, later we can use this to filter outliers
        print(f"[INFO]: Average Inlier reprojection error: {np.mean(inlier_errors)}")
        print(f"[INFO]: Average Outlier reprojection error: {np.mean(outlier_errors)}")

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transformation matrix (T_current_previous)
        # This matrix transforms points from the 'previous' camera frame to the 'current' camera frame
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = tvec.flatten()

        return T_curr_prev

    def is_processing_frame(self) -> bool:
        """
        Check if the SLAM system is currently processing a frame.
        """
        return self.is_processing

    def process_frame_data(self, curr_frame_data: dict, is_threaded: bool = True):
        """
        The function to process the current frame data exposed to the user
        """
        if is_threaded:
            if not self.is_processing:
                self.is_processing = True
                self.process_thread = threading.Thread(
                    target=self._process_frame_async, args=(curr_frame_data, self.data_lock)
                )
                self.process_thread.start()
        else:
            self._process_frame_data(curr_frame_data, self.data_lock)

    def _process_frame_async(self, curr_frame_data: dict, data_lock: threading.Lock):
        """
        Internal function to handle the actual processing in a separate thread
        """
        try:
            self._process_frame_data(curr_frame_data, data_lock)
        except Exception as e:
            print(f"[ERROR]: Failed to process frame data: {e}")
        finally:
            self.is_processing = False

    @timing_decorator
    def _process_frame_data(self, curr_frame_data: dict, data_lock: threading.Lock):
        """
        Internal function to process the frame data
        """
        current_raw_pcd_numpy = curr_frame_data["point_cloud"]
        current_bgr_colors = curr_frame_data["point_bgr_colors"][:, ::-1]

        current_o3d_pcd_with_color = o3d.geometry.PointCloud()
        current_o3d_pcd_with_color.points = o3d.utility.Vector3dVector(current_raw_pcd_numpy)
        current_o3d_pcd_with_color.colors = o3d.utility.Vector3dVector(current_bgr_colors)

        self.T_curr_prev = np.eye(4)
        if self.last_frame_data is None:
            # Protect shared data access with a lock
            with data_lock:
                # First frame: Initialize current map and current camera pose
                current_o3d_pcd_with_color = current_o3d_pcd_with_color.voxel_down_sample(
                    voxel_size=self.current_voxel_size
                )
                self.current_map_pcd = copy.deepcopy(current_o3d_pcd_with_color)

                # Set the initial camera pose to identity or origin of the world
                self.current_camera_pose = np.eye(4)

                self.camera_trajectory_points.append(self.current_camera_pose[:3, 3])

            self.last_frame_data = copy.deepcopy(curr_frame_data)
        else:
            # Subsequent frames: Estimate pose and integrate into map
            # T_curr_prev transforms points from the previous camera frame to the current camera frame.
            T_curr_prev = self._estimate_pose_from_features(self.last_frame_data, curr_frame_data)
            self.T_curr_prev = copy.deepcopy(T_curr_prev)

            if T_curr_prev is not None:
                # Protect shared data access with a lock
                with data_lock:
                    # Update current camera pose: T_world_current = T_world_previous @ T_previous_current
                    # Note that T_previous_current is inv(T_current_previous)
                    self.current_camera_pose = self.current_camera_pose @ np.linalg.inv(T_curr_prev)

                    # Add current camera position to trajectory
                    self.camera_trajectory_points.append(self.current_camera_pose[:3, 3])

                # Transform the current point cloud to the current coordinate system
                transformed_pcd = current_o3d_pcd_with_color.transform(self.current_camera_pose)

                # Protect shared data access with a lock
                with data_lock:
                    self.current_map_pcd = copy.deepcopy(transformed_pcd)

                # Updating last frame only if pose estimation was successful
                self.last_frame_data = copy.deepcopy(curr_frame_data)
            else:
                print("[Error]: Pose estimation failed for given frame. Skipping integration.")

    def get_current_slam_results(self) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """
        Returns the current current map point cloud current calculated information.
        """
        with self.data_lock:
            current_map_pcd = copy.deepcopy(self.current_map_pcd)
            current_camera_pose = copy.deepcopy(self.current_camera_pose)
            T_curr_prev = copy.deepcopy(self.T_curr_prev)

        return current_map_pcd, current_camera_pose, T_curr_prev
