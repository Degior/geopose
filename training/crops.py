"""Select training crops using finite target coverage, including zero heights."""

import math
import random

import numpy as np


def choose_crop(agl, size, min_valid_fraction=0.0, min_valid_pixels=1024,
                attempts=10, rng=random):
    if agl.ndim != 2 or size <= 0 or min(agl.shape) < size:
        raise ValueError(f"Cannot take a {size}x{size} crop from AGL shape {agl.shape}")
    if not 0 <= min_valid_fraction <= 1 or attempts <= 0:
        raise ValueError("Invalid crop coverage or attempt count")
    required = max(min(min_valid_pixels, size * size), math.ceil(size * size * min_valid_fraction))
    best_count = 0
    for _ in range(attempts):
        y = rng.randint(0, agl.shape[0] - size)
        x = rng.randint(0, agl.shape[1] - size)
        valid_count = int(np.count_nonzero(np.isfinite(agl[y:y + size, x:x + size])))
        best_count = max(best_count, valid_count)
        if valid_count >= required:
            return y, x
    raise ValueError(f"No valid crop after {attempts} attempts: best {best_count}/{size * size} "
                     f"finite pixels, required {required}")
