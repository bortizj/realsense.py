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
from rsmodule.capture_simulator import RealSenseCaptureSimulator
from rsmodule.visual_odometry_slam import VisualSLAM
from rsmodule.utils import pad_and_hstack_images


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


class RealSenseBasicCaptureVisualizer:
    def __init__(
        self,
        capture: RealSenseCapture | RealSenseCaptureSimulator,
        min_max_exposure_color: tuple[int, int] = (100, 5000),
        min_max_exposure_depth: tuple[int, int] = (20000, 165000),
    ):
        # identifying if the capture is a simulator or a real camera
        self.read_only = isinstance(capture, RealSenseCaptureSimulator)

        self.exposure_color = min_max_exposure_color[0]
        self.exposure_depth = min_max_exposure_depth[0]
        self.auto_exposure_color = 1
        self.auto_exposure_depth = 1
        self.capture = capture

        self.is_playing = False
        if not self.read_only:
            self.capture.set_exposure(self.exposure_color, self.exposure_depth)
            self.is_playing = True

        self.win_name = "Settings Control"

        self.lock_window_control = False
        self.first_run = True

        self._init_ui(min_max_exposure_color, min_max_exposure_depth)

    def _init_ui(self, min_max_exposure_color, min_max_exposure_depth):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar(
            "Exposure color", self.win_name, self.exposure_color, min_max_exposure_color[1], self._on_exposure_color
        )
        cv2.createTrackbar(
            "Exposure depth", self.win_name, self.exposure_depth, min_max_exposure_depth[1], self._on_exposure_depth
        )
        cv2.createTrackbar(
            "Auto exposure color", self.win_name, self.auto_exposure_color, 1, self._on_auto_exposure_color
        )
        cv2.createTrackbar(
            "Auto exposure depth", self.win_name, self.auto_exposure_depth, 1, self._on_auto_exposure_depth
        )

        data = self.capture.get_frame_data()
        self._update_display(data)

    def __del__(self):
        cv2.destroyAllWindows()

    def _on_auto_exposure_color(self, val):
        self.auto_exposure_color = val
        if not self.lock_window_control and not self.read_only:
            self.capture.set_auto_exposure(self.auto_exposure_color, self.auto_exposure_depth)
            self._on_exposure_color(cv2.getTrackbarPos("Exposure color", self.win_name))

    def _on_auto_exposure_depth(self, val):
        self.auto_exposure_depth = val
        if not self.lock_window_control and not self.read_only:
            self.capture.set_auto_exposure(self.auto_exposure_color, self.auto_exposure_depth)
            self._on_exposure_depth(cv2.getTrackbarPos("Exposure depth", self.win_name))

    def _on_exposure_color(self, val):
        if not self.auto_exposure_color and not self.lock_window_control and not self.read_only:
            self.exposure_color = max(10, val)
            self.capture.set_exposure(self.exposure_color, None)

    def _on_exposure_depth(self, val):
        if not self.auto_exposure_depth and not self.lock_window_control and not self.read_only:
            self.exposure_depth = max(int(10e3), val)
            self.capture.set_exposure(None, self.exposure_depth)

    def _update_display(self, data):
        bgr_img = data["bgr_image"].astype("uint8")
        mono_img = np.dstack((data["ir_left"], data["ir_left"], data["ir_left"])).astype("uint8")
        img = pad_and_hstack_images(bgr_img, mono_img)

        self.img_txt = np.zeros_like(img, dtype="uint8")
        cv2.putText(
            self.img_txt,
            f"Frame id: {self.capture.frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if self.first_run and self.lock_window_control and not self.read_only:
            self.first_run = False
            cv2.putText(
                self.img_txt,
                f"Color Exposure: {self.exposure_color}, Auto: {self.auto_exposure_color}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                self.img_txt,
                f"Depth Exposure: {self.exposure_depth}, Auto: {self.auto_exposure_depth}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        img = cv2.addWeighted(img, 1.0, self.img_txt, 0.5, 0)

        cv2.imshow(self.win_name, img)

    def run(self, lock_window_control: bool = False):
        self.lock_window_control = lock_window_control
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("p") and self.read_only:
                self.is_playing = not self.is_playing

            if self.is_playing:
                if self.lock_window_control and not self.read_only:
                    data = self.capture.get_and_store_frame_data()
                else:
                    data = self.capture.get_frame_data()
                if not data:
                    print("[INFO]: End of data stream")
                    break
                self._update_display(data)
            else:
                if key == ord("a"):
                    data = self.capture.get_frame_data(go_to="previous")
                    if data:
                        self._update_display(data)
                elif key == ord("d"):
                    data = self.capture.get_frame_data(go_to="next")
                    if data:
                        self._update_display(data)


class SLAMOfflineVisualizer:
    def __init__(
        self,
        capture: RealSenseCaptureSimulator,
        slam_system: VisualSLAM,
    ):
        self.capture = capture
        self.slam_system = slam_system

        self.win_name = "SLAM Offline Visualizer"

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def __del__(self):
        cv2.destroyAllWindows()

    def _update_display(self, data):
        bgr_img = data["bgr_image"].astype("uint8")
        mono_img = np.dstack((data["ir_left"], data["ir_left"], data["ir_left"])).astype("uint8")
        img = pad_and_hstack_images(bgr_img, mono_img)

        self.img_txt = np.zeros_like(img, dtype="uint8")
        cv2.putText(
            self.img_txt,
            f"Frame id: {self.capture.frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        img = cv2.addWeighted(img, 1.0, self.img_txt, 0.5, 0)

        cv2.imshow(self.win_name, img)

    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            data = self.capture.get_frame_data()
            self._update_display(data)

            self.slam_system.process_frame_data(data, is_threaded=False)

            print(len(self.slam_system.global_map_pcd.points))
            self.slam_system.camera_trajectory_points
            self.slam_system.current_camera_pose
