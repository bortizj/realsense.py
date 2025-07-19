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

from pathlib import Path

from rsmodule import CameraIntrinsics
from rsmodule.utils import unpickle_from_bytes

import logging

main_logger = logging.getLogger(__name__)


class RealSenseCaptureSimulator:
    """
    Object that simulates the camera by reading from the binary files

    Handy to capture: BGR color images, Depth images, Point clouds, Infrared
    (IR) pattern images (left and right)
    """

    def __init__(
        self,
        path_data: Path,
    ):
        """
        Initializes the capture simulator from stored data.
        """
        self.path_data = path_data
        if not self.path_data.exists():
            main_logger.error("Please provide a path to the data folder.")
            return

        self._read_camera_data()
        self.frame_id = 0

        sn = self.serial_number
        main_logger.info(f"RealSense SN: {sn} simulator initialized with resolution {self.w}x{self.h}.")

    def get_frame_data(self, go_to: str = "") -> tuple[int, dict]:
        """
        Gets the frame data from th binary files

        Returns:
            dict: A dictionary containing the captured data.
                Keys: 'bgr_image', 'depth_image', 'point_cloud', 'ir_left', 'ir_right'.
                Values will be None if a specific data type cannot be retrieved.
        """
        try:
            if go_to == "previous":
                frame_offset = -1
            elif go_to == "next":
                frame_offset = 1
            else:
                frame_offset = 0

            frame_id = self.frame_id + frame_offset
            with open(self.path_data.joinpath("data", f"id_{frame_id}.gz"), "rb") as file:
                data = unpickle_from_bytes(file.read())
                if frame_offset >= 0:
                    self.frame_id += 1
                else:
                    self.frame_id -= 1
        except FileNotFoundError:
            main_logger.error(f"Frame data for id {self.frame_id + frame_offset} not found.")
            return -1, {}

        return frame_id, data

    def get_total_frames(self) -> int:
        """
        Returns the total number of frames available in the data folder.
        """
        return len(list(self.path_data.joinpath("data").glob("id_*.gz")))

    def _read_camera_data(self):
        """
        Reads the camera intrinsics and distortion coefficients in a file
        """
        with open(self.path_data.joinpath("camera_data.gz"), "rb") as file:
            camera_data = unpickle_from_bytes(file.read())
            self.serial_number = camera_data["serial_number"]
            self.dist_coeffs = camera_data["dist_coefficients"]
            self.fx, self.fy, self.cx, self.cy, self.w, self.h = camera_data["intrinsics"]
            main_logger.info(f"Camera Intrinsics: fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}")

        return camera_data

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
