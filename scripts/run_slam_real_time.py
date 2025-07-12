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


if __name__ == "__main__":
    slam_system = VisualSLAM()
    capture = RealSenseCapture(path_store=Path(r"E:\gitProjects\test_folder"))

    # For now infinitely run the SLAM system
    # In the future, this will be replaced with a more sophisticated loop
    merge_count = 0
    while True:
        data = capture.get_and_store_frame_data()
        slam_system.process_frame_data(data, merge_count)

    # Waiting for the store thread to finish if it is alive
    if capture.store_thread and capture.store_thread.is_alive():
        capture.store_thread.join()
