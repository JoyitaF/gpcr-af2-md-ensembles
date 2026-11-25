"""Check that plot artists retain the supplied numerical data."""

import hashlib

import numpy as np


def checksum(array):
    values = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def verify_values(name, actual, expected):
    np.testing.assert_array_equal(actual, expected)
    digest = checksum(actual)
    if digest != checksum(expected):
        raise ValueError(f"{name}: plotted checksum differs from saved values")
    print(f"{name}: {np.size(actual)} plotted values match saved rows; SHA-256 {digest}")
