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

import open3d as o3d
import copy


def combine_point_clouds(
    global_map_pcd: o3d.geometry.PointCloud,
    current_pcd: o3d.geometry.PointCloud,
    current_voxel_size: float = 0.02,
    global_voxel_size: float = 0.05,
    downsample_interval: int = 10,
    merge_count: int = 0,
) -> tuple[o3d.geometry.PointCloud, int]:
    # Down-sampling the current PCD before merging
    if current_voxel_size > 0:
        current_pcd_downsampled = current_pcd.voxel_down_sample(voxel_size=current_voxel_size)
    else:
        current_pcd_downsampled = copy.deepcopy(current_pcd)

    # Merge the current point cloud into the global map
    updated_global_map_pcd = global_map_pcd + current_pcd_downsampled
    merge_count += 1

    # Voxel grid down-sampling the entire global map periodically
    if global_voxel_size > 0 and merge_count % downsample_interval == 0:
        original_size = len(updated_global_map_pcd.points)
        updated_global_map_pcd = updated_global_map_pcd.voxel_down_sample(voxel_size=global_voxel_size)

        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        updated_global_map_pcd.estimate_normals(search_param=search_param)

        updated_size = len(updated_global_map_pcd.points)
        print(f"[INFO]: Global map down-sampled from {original_size} to {updated_size} points.")

    return updated_global_map_pcd, merge_count
