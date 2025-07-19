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

from rsmodule.utils import setup_logging
import logging


if __name__ == "__main__":
    setup_logging(Path(r"E:\gitProjects\test_folder"))
    main_logger = logging.getLogger(__name__)

    capture = RealSenseCapture(width=640, height=480, path_store=Path(r"E:\gitProjects\test_folder"))
    capture.set_exposure(500)
    slam_system = VisualSLAM(capture.get_intrinsics(), dist_coeffs=capture.get_dist_coefficients())
    slam_visualizer = SLAMVisualizer(capture, slam_system)
    slam_visualizer.run()
