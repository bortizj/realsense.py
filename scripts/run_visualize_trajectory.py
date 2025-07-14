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

from rsmodule.utils import read_poses_from_file
from rsmodule.visualization import visualize_trajectories


if __name__ == "__main__":
    in_capture_mode = False
    str_path = r""
    if len(sys.argv) > 1:
        str_path = sys.argv[1]

    if str_path == "":
        print("[ERROR] Please provide a path to the data folder.")
        sys.exit(0)

    poses = read_poses_from_file(Path(str_path))

    visualize_trajectories(poses)
