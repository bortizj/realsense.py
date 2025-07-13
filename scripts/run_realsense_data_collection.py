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
from rsmodule.visualization import SettingControlWindow


if __name__ == "__main__":
    capture = RealSenseCapture(width=640, height=480, path_store=Path(r"E:\gitProjects\test_folder"))

    control = SettingControlWindow(capture)
    control.run()

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
