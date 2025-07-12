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

from rsmodule.capture_module import RealSenseCapture
from rsmodule.visual_odometry_slam import VisualSLAM
from rsmodule.visualization import SLAMVisualizer


if __name__ == "__main__":
    capture = RealSenseCapture(width=640, height=480, path_store=Path(r"E:\gitProjects\test_folder"))
    capture.set_exposure(500)
    slam_system = VisualSLAM(capture.get_intrinsics(), dist_coeffs=capture.get_dist_coefficients())
    slam_visualizer = SLAMVisualizer()

    # For now infinitely run the SLAM system
    # In the future, this will be replaced with a more sophisticated loop
    while True:
        data = capture.get_frame_data()
        slam_system.process_frame_data(data)
        slam_visualizer.update(
            slam_system.global_map_pcd, slam_system.current_camera_pose, slam_system.camera_trajectory_points
        )

    # Waiting for the store thread to finish if it is alive
    if capture.store_thread and capture.store_thread.is_alive():
        capture.store_thread.join()
