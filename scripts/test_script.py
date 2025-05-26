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

from rsmodule.capture_module import RealSenseCapture

from rsmodule.visualization import RealSenseVisualizer

import cv2


if __name__ == "__main__":
    capture = RealSenseCapture()
    visualizer = RealSenseVisualizer()

    while True:
        data = capture.get_frame_data()
        visualizer.update(data)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    visualizer.stop()
    print("Capture module executed successfully.")
