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

import pyrealsense2 as rs
import numpy as np


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

        # Get device and depth sensor to align streams
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Create an align which aligns depth frames to a specific stream type (COLOR in this case)
        self.align = rs.align(rs.stream.color)

        # For point cloud generation
        self.point_cloud = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, dec_magnitude)

        print(f"Info: RealSense pipeline initialized with resolution {width:d}x{height:d} at {fps:d} fps.")

    def get_frame_data(self) -> dict:
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
            "ir_left": None,
            "ir_right": None,
        }

        if color_frame:
            data["bgr_image"] = np.asanyarray(color_frame.get_data())

        if depth_frame:
            # Convert depth to meters
            data["depth_image"] = np.asanyarray(depth_frame.get_data()) * self.depth_scale

            # Generate point cloud
            points = self.point_cloud.calculate(depth_frame)
            vtx = np.asanyarray(points.get_vertices())
            # Reshape to (N, 3) for X, Y, Z coordinates
            if vtx.size > 0:
                data["point_cloud"] = vtx.view(np.float32).reshape(-1, 3)

        if ir_left_frame:
            data["ir_left"] = np.asanyarray(ir_left_frame.get_data())

        if ir_right_frame:
            data["ir_right"] = np.asanyarray(ir_right_frame.get_data())

        return data

    def stop(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()
        print("Info: RealSense pipeline stopped.")
