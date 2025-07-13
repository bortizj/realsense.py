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

from rsmodule.capture_module import RealSenseCapture
from rsmodule.visual_odometry_slam import VisualSLAM


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

    def __init__(self, capture: RealSenseCapture, slam_system: VisualSLAM):
        # The slam for calculations and the capture for data
        self.slam_system = slam_system
        self.capture = capture

        # Open3D Visualizer
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("SLAM Visualizer", 960, 540)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self.__del__)
        self.window.set_on_tick_event(self._update)

        # The widget for the scene
        self.scene_widget = o3d.visualization.gui.SceneWidget()
        self.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene = self.scene
        self.window.add_child(self.scene_widget)

        # Transformation matrix to align the visualizer with the camera coordinate system
        self.transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Default material for rendering
        self.default_material = o3d.visualization.rendering.MaterialRecord()
        self.default_material.shader = "defaultUnlit"

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultUnlit"
        self.pcd_material.point_size = 3.0

        self.line_material = o3d.visualization.rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 2.0

        # Add a coordinate frame for the world origin
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

        # Add a coordinate frame for the current camera pose
        self.camera_frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        self.camera_frame_current = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

        # The line set for the camera trajectory
        self.camera_trajectory_lines = o3d.geometry.LineSet()
        self.camera_trajectory_lines.points = o3d.utility.Vector3dVector()
        self.camera_trajectory_lines.lines = o3d.utility.Vector2iVector()
        self.camera_trajectory_lines.colors = o3d.utility.Vector3dVector()

        # Global map point cloud
        self.global_map_pcd = o3d.geometry.PointCloud()
        self.global_map_pcd.points = o3d.utility.Vector3dVector()
        self.global_map_pcd.colors = o3d.utility.Vector3dVector()
        self.global_map_pcd.normals = o3d.utility.Vector3dVector()

        # Adding the geometries to the visualizer
        self.scene.add_geometry("origin_frame", self.origin_frame, self.default_material)
        self.scene.add_geometry("global_map_pcd", self.global_map_pcd, self.pcd_material)
        self.scene.add_geometry("camera_frame_current", self.camera_frame_current, self.default_material)
        self.scene.add_geometry("camera_trajectory_lines", self.camera_trajectory_lines, self.line_material)

    def __del__(self):
        if self.slam_system.process_thread and self.slam_system.process_thread.is_alive():
            print("Visualizer: Waiting for processing thread to finish...")
            self.slam_system.process_thread.join()
        self.app.quit()
        return True

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene_widget.frame = r

    def _update(self):
        """
        Updates the visualizer with new data
        """
        start_time_pose = time.time()

        # Capture and process the frame data
        data = self.capture.get_frame_data()
        self.slam_system.process_frame_data(data)

        # Safely access the shared data from the SLAM system
        with self.slam_system.data_lock:
            pcd = copy.deepcopy(self.slam_system.global_map_pcd)
            camera_trajectory_points = copy.deepcopy(self.slam_system.camera_trajectory_points)
            camera_pose = copy.deepcopy(self.slam_system.current_camera_pose)

        # Updates the point cloud with the latest data if point cloud has new points
        if len(pcd.points) > len(self.global_map_pcd.points):
            self.global_map_pcd.points = pcd.points
            self.global_map_pcd.colors = pcd.colors
            self.global_map_pcd.normals = pcd.normals

        # Update the LineSet for visualization if there are new trajectory points
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

            # Resetting the camera frame to the base frame
            self.camera_frame_current.vertex_normals = self.camera_frame_base.vertex_normals
            self.camera_frame_current.vertex_colors = self.camera_frame_base.vertex_colors
            self.camera_frame_current.triangle_normals = self.camera_frame_base.triangle_normals
            self.camera_frame_current.triangle_uvs = self.camera_frame_base.triangle_uvs
            self.camera_frame_current.transform(camera_pose)

        self.window.set_needs_layout()

        end_time_pose = time.time()
        print(f"[INFO]: Updating render took: {(end_time_pose - start_time_pose) * 1000:.2f} ms")

        return True

    def run(self):
        """
        Runs the visualizer application
        """
        self.app.run()
