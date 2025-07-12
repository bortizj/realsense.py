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
            print(f"[Error]: Data path {self.path_data} does not exist.")
            return

        self._read_camera_data()
        self.frame_id = 0

        sn = self.serial_number
        print(f"[INFO]: RealSense SN: {sn} simulator initialized with resolution {self.w}x{self.h}.")

    def get_frame_data(self) -> dict:
        """
        Gets the frame data from th binary files

        Returns:
            dict: A dictionary containing the captured data.
                Keys: 'bgr_image', 'depth_image', 'point_cloud', 'ir_left', 'ir_right'.
                Values will be None if a specific data type cannot be retrieved.
        """
        try:
            with open(self.path_data.joinpath(f"id_{self.frame_id}.gz"), "rb") as file:
                data = unpickle_from_bytes(file.read())
                self.frame_id += 1
        except FileNotFoundError:
            print(f"[Error]: Frame data for id {self.frame_id} not found.")
            return {}

        return data

    def _read_camera_data(self):
        """
        Reads the camera intrinsics and distortion coefficients in a file
        """
        with open(self.path_data.joinpath("camera_data.gz"), "rb") as file:
            camera_data = unpickle_from_bytes(file.read())
            self.serial_number = camera_data["serial_number"]
            self.dist_coeffs = camera_data["dist_coefficients"]
            self.fx, self.fy, self.cx, self.cy, self.w, self.h = camera_data["intrinsics"]
            print(f"[INFO]: Camera Intrinsics: fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}")

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
