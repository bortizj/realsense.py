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
