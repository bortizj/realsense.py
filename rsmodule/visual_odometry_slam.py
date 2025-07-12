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
import time

import copy

from rsmodule.capture_module import RealSenseCapture
from rsmodule.o3d_processing import combine_point_clouds


class VisualSLAM:
    """
    Object to perform Visual SLAM using Intel RealSense camera data.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        nmatches: int = 200,
        global_voxel_size: float = 0.005,
        current_voxel_size: float = 0.005,
    ):
        # Initialize the RealSense camera capture
        self.nmatches = nmatches
        self.width = width
        self.height = height
        self.fps = fps
        self.global_voxel_size = global_voxel_size
        self.current_voxel_size = current_voxel_size
        self.capture = RealSenseCapture(width=width, height=height, fps=fps)

        # Camera Intrinsics (will be populated from RealSense)
        self._get_realsense_intrinsics()

        # Global map point cloud
        self.global_map_pcd = o3d.geometry.PointCloud()
        self.current_camera_pose = np.eye(4)

        # Open3D Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="RealSense SLAM", width=self.width, height=self.height)

        # Get the view control
        self.view_ctl = self.vis.get_view_control()

        # Set the camera to look at the center of the object
        self.view_ctl.set_lookat([0, 0, 0])

        # Add a coordinate frame for the world origin
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(self.origin_frame)
        self.vis.add_geometry(self.global_map_pcd)

        # Add a coordinate frame for the current camera pose
        self.camera_frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.vis.add_geometry(self.camera_frame)

        # For camera trajectory visualization
        self.camera_trajectory_points = []
        self.camera_trajectory_lines = o3d.geometry.LineSet()
        self.vis.add_geometry(self.camera_trajectory_lines)

        # Feature detector ORB and brute force matcher for the visual odometry
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        print("[INFO]: Visual SLAM system initialized.")

    def _get_realsense_intrinsics(self):
        """
        Retrieves camera intrinsics from the RealSense pipeline
        """
        self.dist_coeffs = self.capture.get_dist_coefficients()
        cam_int = self.capture.get_intrinsics()
        self.fx, self.fy, self.cx, self.cy, self.width, self.height = cam_int
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

        print(f"[INFO]: Camera Intrinsics: {self.intrinsic}")

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

    def _estimate_pose_from_features(self, prev_frame_data: dict, curr_frame_data: dict) -> np.ndarray:
        """
        Estimates the relative pose (T_curr_prev) from previous to current frame
        using feature matching and RANSAC PnP.
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
            if p3d[2] > 0.01 and p3d[2] < 10.0:
                points3D_prev.append(p3d)
                points2D_curr.append((u2, v2))

        if len(points3D_prev) < 4:
            print("[Error]: Not enough valid 3D-2D correspondences for PnP.")
            return None

        points3D_prev = np.array(points3D_prev, dtype=np.float32)
        points2D_curr = np.array(points2D_curr, dtype=np.float32)

        # Camera matrix (intrinsics) for OpenCV PnP and the distortion coefficients
        camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = self.dist_coeffs.reshape((1, 5))

        # Solve PnP using RANSAC for robust estimation
        # rvec: rotation vector (Rodrigues), tvec: translation vector
        # inliers: mask of inlier points
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3D_prev,
            imagePoints=points2D_curr,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            reprojectionError=3.0,
            iterationsCount=100,
        )

        if not success:
            print("[Error]: PnP failed to estimate pose.")
            return None

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transformation matrix (T_current_previous)
        # This matrix transforms points from the 'previous' camera frame to the 'current' camera frame
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = tvec.flatten()

        return T_curr_prev

    def run(self):
        last_frame_data = None
        frame_idx = 0
        merge_count = 0

        while True:
            curr_frame_data = self.capture.get_frame_data()
            bgr_image = curr_frame_data["bgr_image"]
            depth_image_meters = curr_frame_data["depth_image"]
            current_raw_pcd_numpy = curr_frame_data["point_cloud"]  # The (N, 3) numpy array point cloud

            if bgr_image is None or depth_image_meters is None or current_raw_pcd_numpy is None:
                print("Skipping frame: Missing data.")
                continue

            # Create RGBD Image to get colors
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            depth_image_mm = (depth_image_meters * 1000).astype(np.uint16)

            o3d_color_img = o3d.geometry.Image(rgb_image)
            o3d_depth_img = o3d.geometry.Image(depth_image_mm)

            o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color_img, o3d_depth_img, depth_scale=1000.0, convert_rgb_to_intensity=False
            )

            # Generate PCD from RGBDImage for accurate color-point mapping
            current_o3d_pcd_with_color = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd_image, self.intrinsic)
            # Filter out invalid points
            current_o3d_pcd_with_color = current_o3d_pcd_with_color.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )[0]
            current_o3d_pcd_with_color.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

            transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            current_o3d_pcd_with_color.transform(transform_matrix)

            if frame_idx == 0:
                # First frame: Initialize global map and current camera pose
                self.global_map_pcd.points = current_o3d_pcd_with_color.points
                self.global_map_pcd.colors = current_o3d_pcd_with_color.colors
                # Set the initial camera pose to identity or origin of the world
                self.current_camera_pose = np.eye(4)
                self.vis.update_geometry(self.global_map_pcd)

                self.camera_trajectory_points.append(self.current_camera_pose[:3, 3])

                print("[INFO]: Initialized global map with first frame")
            else:
                # Subsequent frames: Estimate pose and integrate into map
                start_time_pose = time.time()
                # T_curr_prev transforms points from the previous camera frame to the current camera frame.
                T_curr_prev = self._estimate_pose_from_features(last_frame_data, curr_frame_data)
                end_time_pose = time.time()
                print(f"[INFO]: Pose estimation took: {(end_time_pose - start_time_pose) * 1000:.2f} ms")

                if T_curr_prev is not None:
                    # Update global camera pose: T_world_current = T_world_previous @ T_previous_current
                    # Note that T_previous_current is inv(T_current_previous)
                    self.current_camera_pose = self.current_camera_pose @ np.linalg.inv(T_curr_prev)

                    # Add current camera position to trajectory
                    self.camera_trajectory_points.append(self.current_camera_pose[:3, 3])

                    # Update the LineSet for visualization
                    if len(self.camera_trajectory_points) > 1:
                        points = np.array(self.camera_trajectory_points)
                        lines = []
                        colors = []
                        for i in range(len(points) - 1):
                            lines.append([i, i + 1])
                            colors.append([1, 0, 0])

                        self.camera_trajectory_lines.points = o3d.utility.Vector3dVector(points)
                        self.camera_trajectory_lines.lines = o3d.utility.Vector2iVector(np.asarray(lines))
                        self.camera_trajectory_lines.colors = o3d.utility.Vector3dVector(np.asarray(colors))

                    # Transform the current point cloud to the global coordinate system
                    transformed_pcd = current_o3d_pcd_with_color.transform(self.current_camera_pose)

                    # Merge with global map
                    merged_pcd, merge_count = combine_point_clouds(
                        self.global_map_pcd,
                        transformed_pcd,
                        merge_count=merge_count,
                        global_voxel_size=self.global_voxel_size,
                        current_voxel_size=self.current_voxel_size,
                    )

                    # Updating the data from the point cloud for visualization this is necessary
                    self.global_map_pcd.clear()
                    self.global_map_pcd.points = merged_pcd.points
                    self.global_map_pcd.colors = merged_pcd.colors
                    self.global_map_pcd.normals = merged_pcd.normals

                    # Copy the points, triangles, colors, and normals from the freshly transformed base
                    transformed_base_frame = copy.deepcopy(self.camera_frame_base)
                    transformed_base_frame.transform(self.current_camera_pose)
                    self.camera_frame.vertices = transformed_base_frame.vertices
                    self.camera_frame.triangles = transformed_base_frame.triangles
                    if transformed_base_frame.has_vertex_normals():
                        self.camera_frame.vertex_normals = transformed_base_frame.vertex_normals
                    else:
                        self.camera_frame.vertex_normals = o3d.utility.Vector3dVector()

                    if transformed_base_frame.has_vertex_colors():
                        self.camera_frame.vertex_colors = transformed_base_frame.vertex_colors
                    else:
                        self.camera_frame.vertex_colors = o3d.utility.Vector3dVector()

                    self.vis.update_geometry(self.global_map_pcd)
                    self.vis.update_geometry(self.camera_frame)
                    self.vis.update_geometry(self.camera_trajectory_lines)
                else:
                    print(f"[Error]: Pose estimation failed for frame {frame_idx}. Skipping integration.")

            self.vis.poll_events()
            self.vis.update_renderer()

            last_frame_data = copy.deepcopy(curr_frame_data)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self.capture.stop()
        self.vis.destroy_window()
        print("[INFO]: SLAM process finished.")
