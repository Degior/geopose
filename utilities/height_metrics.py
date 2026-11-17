"""Per-image AGL metrics matching Depth Anything's metric_depth/util/metric.py."""

import numpy as np


METRIC_NAMES = (
    "abs_rel", "bias", "d1", "d2", "d3", "log10", "mae", "rmse",
    "rmse_log", "silog", "sq_rel",
)


def evaluate_height(prediction, target, max_height=250.0, eps=1e-6):
    """Return metrics and pixel counts for one image, with heights in metres.

    Absolute metrics include valid zero heights. Ratio/log metrics use target
    heights > eps and predictions clipped below at eps. SILog deliberately uses
    the same 0.5 coefficient and no factor of 100 as the comparison project.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} / {target.shape}")
    if not np.isfinite(max_height) or max_height <= 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("max_height and eps must be positive and finite")
    valid_target = np.isfinite(target) & (target >= 0) & (target <= max_height)
    nonfinite_predictions = int(np.count_nonzero(valid_target & ~np.isfinite(prediction)))
    valid = valid_target & np.isfinite(prediction)
    prediction, target = prediction[valid], target[valid]
    counts = {
        "valid_pixels": int(target.size),
        "positive_pixels": int(np.count_nonzero(target > eps)),
        "nonfinite_predictions": nonfinite_predictions,
    }
    if not target.size:
        raise ValueError("Image has no valid pixels for evaluation")
    difference = prediction - target
    metrics = {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "bias": float(np.mean(difference)),
    }
    positive = target > eps
    if np.any(positive):
        pred = np.maximum(prediction[positive], eps)
        truth = target[positive]
        difference = pred - truth
        log_difference = np.log(pred) - np.log(truth)
        ratio = np.maximum(truth / pred, pred / truth)
        metrics.update({
            "d1": float(np.mean(ratio < 1.25)),
            "d2": float(np.mean(ratio < 1.25 ** 2)),
            "d3": float(np.mean(ratio < 1.25 ** 3)),
            "abs_rel": float(np.mean(np.abs(difference) / truth)),
            "sq_rel": float(np.mean(difference ** 2 / truth)),
            "rmse_log": float(np.sqrt(np.mean(log_difference ** 2))),
            "log10": float(np.mean(np.abs(np.log10(pred) - np.log10(truth)))),
            "silog": float(np.sqrt(max(
                0.0, np.mean(log_difference ** 2) - 0.5 * np.mean(log_difference) ** 2,
            ))),
        })
    return metrics, counts


class HeightMetricAverage:
    """Average each metric over images where that metric is defined."""

    def __init__(self):
        self.sums = {name: 0.0 for name in METRIC_NAMES}
        self.counts = {name: 0 for name in METRIC_NAMES}
        self.images = 0
        self.pixels = {name: 0 for name in (
            "valid_pixels", "positive_pixels", "nonfinite_predictions",
        )}

    def update(self, metrics, counts):
        for name, value in metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"Non-finite metric: {name}={value}")
            self.sums[name] += value
            self.counts[name] += 1
        self.images += 1
        for name in self.pixels:
            self.pixels[name] += counts[name]

    def report(self):
        return {
            "metrics": {
                name: self.sums[name] / self.counts[name] if self.counts[name] else None
                for name in METRIC_NAMES
            },
            "images": self.images,
            "metric_image_counts": dict(self.counts),
            **self.pixels,
        }
