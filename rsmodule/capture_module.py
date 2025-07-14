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

import time

import pyrealsense2 as rs
import threading
import numpy as np

from pathlib import Path

from rsmodule import CameraIntrinsics
from rsmodule.utils import pickle_to_bytes


class RealSenseCapture:
    """
    Object to capture the various data types from an Intel RealSense camera.

    Handy to capture: BGR color images, Depth images, Point clouds, Infrared
    (IR) pattern images (left and right)
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        dec_magnitude: int = 2,
        path_store: Path | None = None,
    ):
        """
        Initializes the RealSense pipeline with specified resolution and frame rate.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Configure depth stream
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Configure left and right infrared streams, respectively
        self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)

        # Get device, color depth sensor to align streams
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.color_sensor = self.device.first_color_sensor()

        # Getting the serial number of the camera
        self.serial_number = self.device.get_info(rs.camera_info.serial_number)

        # Create an align which aligns depth frames to a specific stream type (COLOR in this case)
        self.align = rs.align(rs.stream.color)

        # For point cloud generation
        self.point_cloud = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, dec_magnitude)

        # Getting the camera intrinsics and distortion coefficients
        self.compute_intrinsics_and_dist_coefficients()

        # The path to store captured data
        self.frame_id = 0
        self.is_storing = False
        self.path_store = path_store
        if self.path_store is not None:
            if not self.path_store.exists():
                self.path_store = None
                print(f"[Error]: Data path {self.path_store} does not exist. Data will not be stored.")

        self.store_camera_data()

        sn = self.serial_number
        print(f"[INFO]: RealSense SN: {sn} initialized with resolution {width}x{height} at {fps} FPS.")

    def __del__(self):
        self.stop()

    def set_auto_exposure(self, auto_color_exposure: bool = True, auto_mono_exposure: bool = True):
        self.color_sensor.set_option(rs.option.enable_auto_exposure, auto_color_exposure)
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, auto_mono_exposure)

        # Given time to apply the settings
        time.sleep(0.05)

    def set_exposure(self, manual_color_exposure: int | None = None, manual_mono_exposure: int | None = None):
        """
        Sets the exposure for both depth and color sensors
        """
        if manual_color_exposure is not None:
            self.color_sensor.set_option(rs.option.exposure, manual_color_exposure)
        if manual_mono_exposure is not None:
            self.depth_sensor.set_option(rs.option.exposure, manual_mono_exposure)

        # Given time to apply the settings
        time.sleep(0.05)

    def get_frame_data(self) -> tuple[int, dict]:
        """
        Captures and returns the latest frames from the RealSense camera.
        This method retrieves the BGR color image, depth image, point cloud data, and infrared images.

        Returns:
            dict: A dictionary containing the captured data.
                Keys: 'bgr_image', 'depth_image', 'point_cloud', 'ir_left', 'ir_right'.
                Values will be None if a specific data type cannot be retrieved.
        """
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get individual frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)

        data = {
            "bgr_image": None,
            "depth_image": None,
            "point_cloud": None,
            "point_bgr_colors": None,
            "ir_left": None,
            "ir_right": None,
        }

        if color_frame:
            data["bgr_image"] = np.asanyarray(color_frame.get_data())
        else:
            print("[Warning]: No color frame captured.")

        if depth_frame:
            # Convert depth to meters
            data["depth_image"] = np.asanyarray(depth_frame.get_data()) * self.depth_scale

            # Generate point cloud
            points = self.point_cloud.calculate(depth_frame)
            vtx = np.asanyarray(points.get_vertices())
            # Reshape to (N, 3) for X, Y, Z coordinates
            if vtx.size > 0:
                data["point_cloud"] = vtx.view(np.float32).reshape(-1, 3)
                data["point_bgr_colors"] = data["bgr_image"].reshape(-1, 3) / 255.0
        else:
            print("[Warning]: No depth frame captured.")

        if ir_left_frame:
            data["ir_left"] = np.asanyarray(ir_left_frame.get_data())
        else:
            print("[Warning]: No left infrared frame captured.")

        if ir_right_frame:
            data["ir_right"] = np.asanyarray(ir_right_frame.get_data())
        else:
            print("[Warning]: No right infrared frame captured.")

        return -1, data

    def get_and_store_frame_data(self) -> tuple[int, dict]:
        """
        Captures the latest frames and stores them in a file
        """
        current_frame_id, data = self.get_frame_data()
        if self.path_store is not None:
            if not self.is_storing:
                self.is_storing = True
                current_frame_id = self.frame_id
                path_store = self.path_store
                path_store.joinpath("data").mkdir(parents=True, exist_ok=True)
                self.frame_id += 1
                self.store_thread = threading.Thread(
                    target=self._store_data_async, args=(data, current_frame_id, path_store)
                )
                self.store_thread.start()
            else:
                print("[INFO]: Already storing data. Skipping this frame.")

        return current_frame_id, data

    def _store_data_async(self, data, current_frame_id, path_store):
        """
        Internal function to handle the actual storage in a separate thread
        """
        try:
            bytes_data = pickle_to_bytes(data)
            with open(path_store.joinpath("data", f"id_{current_frame_id}.gz"), "wb") as file:
                file.write(bytes_data)
        except Exception as e:
            print(f"[ERROR]: Failed to store frame data id_{current_frame_id}.gz: {e}")
        finally:
            self.is_storing = False
            print(f"[INFO]: Frame data stored as id_{current_frame_id}.gz")

    def store_camera_data(self):
        """
        Stores the camera intrinsics and distortion coefficients in a file
        """
        if self.path_store is not None:
            camera_data = {
                "intrinsics": self.get_intrinsics(),
                "dist_coefficients": self.get_dist_coefficients(),
                "serial_number": self.get_serial_number(),
            }
            bytes_data = pickle_to_bytes(camera_data)
            with open(self.path_store.joinpath("camera_data.gz"), "wb") as file:
                file.write(bytes_data)
                print("[INFO]: Camera data stored successfully.")

    def compute_intrinsics_and_dist_coefficients(self):
        """
        Retrieves camera intrinsics from the RealSense pipeline
        """
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()

        # Getting camera intrinsics and distortion coefficients
        self.dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
        self.fx, self.fy = intrinsics.fx, intrinsics.fy
        self.cx, self.cy = intrinsics.ppx, intrinsics.ppy
        self.w, self.h = intrinsics.width, intrinsics.height

        print(f"[INFO]: Camera Intrinsics: fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}")

    def get_intrinsics(self):
        """
        Returns the camera intrinsics
        """
        return CameraIntrinsics(self.fx, self.fy, self.cx, self.cy, self.w, self.h)

    def get_serial_number(self):
        """
        Returns the serial number of the camera
        """
        return self.serial_number

    def get_dist_coefficients(self):
        """
        Returns the distortion coefficients of the camera
        """
        return self.dist_coeffs

    def stop(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()
        print("Info: RealSense pipeline stopped.")
