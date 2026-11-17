"""Select training crops using finite target coverage, including zero heights."""

import math
import random

import numpy as np


def choose_crop(agl, size, min_valid_fraction=0.0, min_valid_pixels=1024,
                attempts=10, rng=random):
    """Prefer the requested coverage, otherwise return the best usable attempt.

    Augmentation and source nodata can make the preferred fraction impossible.
    The hard floor is min_valid_pixels, not the preferred coverage fraction.
    """
    if agl.ndim != 2 or size <= 0 or min(agl.shape) < size:
        raise ValueError(f"Cannot take a {size}x{size} crop from AGL shape {agl.shape}")
    if not 0 <= min_valid_fraction <= 1 or attempts <= 0 or min_valid_pixels < 1:
        raise ValueError("Invalid crop coverage or attempt count")
    minimum = min(min_valid_pixels, size * size)
    preferred = max(minimum, math.ceil(size * size * min_valid_fraction))
    best_count = 0
    best_crop = None
    for _ in range(attempts):
        y = rng.randint(0, agl.shape[0] - size)
        x = rng.randint(0, agl.shape[1] - size)
        valid_count = int(np.count_nonzero(np.isfinite(agl[y:y + size, x:x + size])))
        if valid_count > best_count:
            best_count, best_crop = valid_count, (y, x)
        if valid_count >= preferred:
            return y, x
    if best_count >= minimum:
        return best_crop
    raise ValueError(f"No valid crop after {attempts} attempts: best {best_count}/{size * size} "
                     f"finite pixels, hard minimum {minimum}")
