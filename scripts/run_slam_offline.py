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

import sys
from pathlib import Path

from rsmodule.capture_simulator import RealSenseCaptureSimulator
from rsmodule.visual_odometry_slam import VisualSLAM
from rsmodule.visualization import SLAMOfflineVisualizer
from rsmodule.utils import setup_logging
import logging


def offline_slam_worker():
    """
    Main function to run the SLAM system in offline mode.
    """
    pass


if __name__ == "__main__":
    in_capture_mode = False
    str_path = r""
    if len(sys.argv) > 1:
        str_path = sys.argv[1]

    setup_logging(Path(str_path))
    main_logger = logging.getLogger(__name__)

    if str_path == "":
        main_logger.error("Please provide a path to the data folder.")
        sys.exit(0)

    main_logger.info("SLAM Offline started.")

    capture = RealSenseCaptureSimulator(Path(str_path))
    slam_system = VisualSLAM(capture.get_intrinsics(), dist_coeffs=capture.get_dist_coefficients())

    control = SLAMOfflineVisualizer(capture, slam_system)
    control.run()
