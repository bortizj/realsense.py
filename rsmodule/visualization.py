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
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="SLAM Visualizer")

        # The point cloud for the global map
        self.global_map_pcd = o3d.geometry.PointCloud()

        # Transformation matrix to align the visualizer with the camera coordinate system
        self.transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Add a coordinate frame for the world origin
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.origin_frame.transform(self.transform_matrix)

        # Add a coordinate frame for the current camera pose
        self.camera_frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.camera_frame_current = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

        # The line set for the camera trajectory
        self.camera_trajectory_lines = o3d.geometry.LineSet()

        # Adding the geometries to the visualizer
        self.vis.add_geometry(self.origin_frame)
        self.vis.add_geometry(self.global_map_pcd)
        self.vis.add_geometry(self.camera_frame_current)
        self.vis.add_geometry(self.camera_trajectory_lines)

    def __del__(self):
        self.vis.destroy_window()

    def update(self, pcd: o3d.geometry.PointCloud, camera_pose: np.ndarray, camera_trajectory_points: list):
        """
        Updates the visualizer with new data
        """
        # Updating the data from the point cloud for visualization this is necessary
        self.global_map_pcd.clear()
        self.global_map_pcd.points = pcd.points
        self.global_map_pcd.colors = pcd.colors
        self.global_map_pcd.normals = pcd.normals

        # Updating the camera frame
        transformed_base_frame = copy.deepcopy(self.camera_frame_base)
        transformed_base_frame.transform(self.transform_matrix)
        transformed_base_frame.transform(camera_pose)
        self.camera_frame_current.vertices = transformed_base_frame.vertices
        self.camera_frame_current.triangles = transformed_base_frame.triangles
        if transformed_base_frame.has_vertex_normals():
            self.camera_frame_current.vertex_normals = transformed_base_frame.vertex_normals
        else:
            self.camera_frame_current.vertex_normals = o3d.utility.Vector3dVector()

        if transformed_base_frame.has_vertex_colors():
            self.camera_frame_current.vertex_colors = transformed_base_frame.vertex_colors
        else:
            self.camera_frame_current.vertex_colors = o3d.utility.Vector3dVector()

        # Update the LineSet for visualization
        if len(camera_trajectory_points) > 1:
            points = np.array(camera_trajectory_points)
            lines = []
            colors = []
            for i in range(len(points) - 1):
                lines.append([i, i + 1])
                colors.append([1, 0, 0])

            self.camera_trajectory_lines.points = o3d.utility.Vector3dVector(points)
            self.camera_trajectory_lines.lines = o3d.utility.Vector2iVector(np.asarray(lines))
            self.camera_trajectory_lines.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        # Update the visualizer with the new geometries
        self.global_map_pcd.transform(self.transform_matrix)
        self.camera_trajectory_lines.transform(self.transform_matrix)

        self.vis.update_geometry(self.global_map_pcd)
        self.vis.update_geometry(self.camera_frame_current)
        self.vis.update_geometry(self.camera_trajectory_lines)
        self.vis.poll_events()
        self.vis.update_renderer()
