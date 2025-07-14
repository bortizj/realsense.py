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

from rsmodule.capture_module import RealSenseCapture
from rsmodule.capture_simulator import RealSenseCaptureSimulator
from rsmodule.visualization import RealSenseBasicCaptureVisualizer


if __name__ == "__main__":
    in_capture_mode = True
    str_path = r""
    if len(sys.argv) > 1:
        str_path = sys.argv[1]
        if len(sys.argv) > 2:
            in_capture_mode = False if sys.argv[2] == "0" else True

    if str_path == "":
        path_store = None
    else:
        path_store = Path(str_path)

    if in_capture_mode:
        capture = RealSenseCapture(width=640, height=480, path_store=path_store)
    else:
        capture = RealSenseCaptureSimulator(Path(str_path))

    control = RealSenseBasicCaptureVisualizer(capture)
    control.run()

    # Only rus the capture module if not in read-only mode
    if not control.read_only:
        print(f"Setting the following parameters to camera Serial number: {capture.get_serial_number()}")
        print(f"Color auto exposure {control.auto_exposure_color}, Mono auto exposure {control.auto_exposure_depth}")
        print(f"Color exposure {control.exposure_color}, Mono exposure {control.exposure_depth}")

        capture.set_auto_exposure(control.auto_exposure_color, control.auto_exposure_depth)
        if control.auto_exposure_color:
            exposure_color = None
        else:
            exposure_color = control.exposure_color
        if control.auto_exposure_depth:
            exposure_depth = None
        else:
            exposure_depth = control.exposure_depth
        capture.set_exposure(exposure_color, exposure_depth)

        control.run(lock_window_control=True)
