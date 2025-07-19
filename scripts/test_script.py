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

import cv2
from pathlib import Path

from rsmodule.capture_simulator import RealSenseCaptureSimulator
from rsmodule.visualization import RealSenseVisualizer
from rsmodule.utils import setup_logging
import logging


if __name__ == "__main__":
    setup_logging(Path(r"E:\gitProjects\test_folder"))
    main_logger = logging.getLogger(__name__)

    capture = RealSenseCaptureSimulator(path_data=Path(r"E:\gitProjects\test_folder"))
    visualizer = RealSenseVisualizer()

    main_logger.info(f"Camera Intrinsics: {capture.get_intrinsics()}")
    main_logger.info(f"Camera Serial Number: {capture.get_serial_number()}")
    main_logger.info(f"Camera Distortion Coefficients: {capture.get_dist_coefficients()}")

    while True:
        __, data = capture.get_frame_data()
        # data = capture.get_and_store_frame_data()
        if not data:
            main_logger.warning("Probably end of data reached.")
            break
        visualizer.update(data)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    visualizer.stop()
    main_logger.info("Capture module executed successfully.")
