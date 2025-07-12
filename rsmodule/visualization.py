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

import open3d as o3d
import numpy as np
import copy
import cv2
import time


DEVICE = o3d.core.Device("cuda:0")


class RealSenseVisualizer:
    """
    Convenient class to visualize images and point clouds from a RealSense camera.
    """

    def __init__(self):
        self.cv_window_names = ["bgr_image", "depth_image", "ir_left", "ir_right"]

        # Create OpenCV windows
        for name in self.cv_window_names:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud", width=1280, height=720)
        self.point_cloud_initialized = False
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_color = o3d.geometry.PointCloud()

    def __del__(self):
        """
        Destroys all windows
        """
        cv2.destroyAllWindows()
        self.vis.destroy_window()
        print("Info: All visualization windows destroyed.")

    def update(self, data: dict):
        """
        Updates the visualizer with new data
        """
        for ii in self.cv_window_names:
            if data[ii] is not None:
                img = data[ii]
                if ii == "depth_image":
                    img = cv2.applyColorMap(cv2.convertScaleAbs(data["depth_image"], alpha=255 / 4.0), cv2.COLORMAP_JET)
                cv2.imshow(ii, img)

        if data["point_cloud"] is not None:
            self.pcd.points = o3d.utility.Vector3dVector(data["point_cloud"])

            transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcd.transform(transform_matrix)

            colors = data["bgr_image"].reshape(-1, 3) / 255.0
            self.pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])

            if self.point_cloud_initialized:
                self.vis.update_geometry(self.pcd)
            else:
                self.vis.add_geometry(self.pcd)
                self.point_cloud_initialized = True

        self.vis.poll_events()
        self.vis.update_renderer()

        cv2.waitKey(1)

    def stop(self):
        self.__del__()


class SLAMVisualizer:
    """
    Convenient class to visualize the point cloud from a SLAM system.
    """

    def __init__(self):
        # Open3D Visualizer
        o3d.visualization.gui.Application.instance.initialize()
        self.vis = o3d.visualization.O3DVisualizer("SLAM Visualizer", 960, 540)
        self.vis.show_settings = True

        # Transformation matrix to align the visualizer with the camera coordinate system
        self.transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Add a coordinate frame for the world origin
        self.origin_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

        # Add a coordinate frame for the current camera pose
        self.camera_frame_base = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.camera_frame_current = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

        # The line set for the camera trajectory
        self.camera_trajectory_lines = o3d.t.geometry.LineSet()

        # Adding the geometries to the visualizer
        self.vis.add_geometry("origin_frame", self.origin_frame)
        self.vis.add_geometry("global_map_pcd", o3d.t.geometry.PointCloud())
        self.vis.add_geometry("camera_frame_current", self.camera_frame_current)
        self.vis.add_geometry("camera_trajectory_lines", self.camera_trajectory_lines)

        o3d.visualization.gui.Application.instance.add_window(self.vis)
        # TODO: We need multi threading as well for the visualizer to avoid blocking the main thread
        o3d.visualization.gui.Application.instance.run()

    def __del__(self):
        self.vis.destroy_window()

    def update(self, pcd: o3d.t.geometry.PointCloud, camera_pose: np.ndarray, camera_trajectory_points: list):
        """
        Updates the visualizer with new data
        """
        start_time_pose = time.time()

        # Update the LineSet for visualization
        if len(camera_trajectory_points) > 1:
            points = np.array(camera_trajectory_points)
            lines = []
            colors = []
            for i in range(len(points) - 1):
                lines.append([i, i + 1])
                colors.append([1, 0, 0])

            self.camera_trajectory_lines.points = o3d.core.Tensor(np.asarray(points), o3d.core.Dtype.Float32, DEVICE)
            self.camera_trajectory_lines.lines = o3d.core.Tensor(np.asarray(lines), o3d.core.Dtype.Float32, DEVICE)
            self.camera_trajectory_lines.colors = o3d.core.Tensor(np.asarray(colors), o3d.core.Dtype.Float32, DEVICE)

        self.camera_frame_current = copy.deepcopy(self.camera_frame_base)
        self.camera_frame_current.transform(camera_pose)

        self.vis.scene.update_geometry("global_map_pcd", pcd)
        self.vis.scene.update_geometry("camera_frame_current", self.camera_frame_current)
        self.vis.scene.update_geometry("camera_trajectory_lines", self.camera_trajectory_lines)

        self.vis.poll_events()
        self.vis.update_renderer()

        end_time_pose = time.time()
        print(f"[INFO]: Updating render took: {(end_time_pose - start_time_pose) * 1000:.2f} ms")
