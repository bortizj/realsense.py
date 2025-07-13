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

import json
import gzip
import pickle
import cv2
import numpy as np


def compress_dict(data_dict: dict) -> bytes:
    """
    Serializes a dictionary to JSON and then compresses it using gzip
    """
    json_data = json.dumps(data_dict, ensure_ascii=False).encode("utf-8")
    compressed_data = gzip.compress(json_data)

    return compressed_data


def decompress_dict(compressed_data: bytes) -> dict:
    """
    Decompresses gzipped data and then deserializes it from JSON
    """
    decompressed_data = gzip.decompress(compressed_data)
    data_dict = json.loads(decompressed_data.decode("utf-8"))

    return data_dict


def pickle_to_bytes(data_dict: dict) -> bytes:
    """
    Pickles a dictionary into bytes
    """
    pickled_data = pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL)

    return pickled_data


def unpickle_from_bytes(pickled_data):
    """
    Unpickles data from a bytes object back into a dictionary
    """
    data_dict = pickle.loads(pickled_data)

    return data_dict


def pad_and_hstack_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Pads two images to the same height and horizontally stacks them.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 > h2:
        # Pad img2 to match h1
        pad_amount = h1 - h2
        # (top, bottom, left, right)
        img2_padded = cv2.copyMakeBorder(img2, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img1_padded = img1
    elif h2 > h1:
        # Pad img1 to match h2
        pad_amount = h2 - h1
        img1_padded = cv2.copyMakeBorder(img1, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img2_padded = img2
    else:
        # Heights are already equal, no padding needed
        img1_padded = img1
        img2_padded = img2

    return np.hstack((img1_padded, img2_padded))
