import csv
import json
import math
import sys
import types

import numpy as np
import pytest

import evaluate_height as runner
from utilities.height_metrics import HeightMetricAverage, evaluate_height


def test_perfect_predictions_include_zero_heights():
    metrics, counts = evaluate_height([0, 1, 2], [0, 1, 2])
    assert counts == {"valid_pixels": 3, "positive_pixels": 2, "nonfinite_predictions": 0}
    for name, value in metrics.items():
        assert value == (1.0 if name in {"d1", "d2", "d3"} else 0.0)


def test_known_scale_error_and_silog_definition():
    metrics, _ = evaluate_height([2, 4], [1, 2])
    assert metrics == pytest.approx({
        "mae": 1.5, "rmse": math.sqrt(2.5), "bias": 1.5,
        "d1": 0, "d2": 0, "d3": 0,
        "abs_rel": 1, "sq_rel": 1.5, "rmse_log": math.log(2),
        "log10": math.log10(2), "silog": math.log(2) / math.sqrt(2),
    })


def test_ratio_thresholds_are_strict():
    metrics, _ = evaluate_height([1.25], [1])
    assert (metrics["d1"], metrics["d2"], metrics["d3"]) == (0, 1, 1)


def test_zero_targets_only_affect_absolute_metrics():
    metrics, counts = evaluate_height([10, 2], [0, 2])
    assert metrics["mae"] == 5
    assert metrics["rmse"] == pytest.approx(math.sqrt(50))
    assert metrics["abs_rel"] == 0
    assert metrics["d1"] == 1
    assert counts["positive_pixels"] == 1


def test_target_range_nodata_and_nonfinite_prediction_mask():
    metrics, counts = evaluate_height([1, 20, 20, 20, np.inf], [1, -1, 251, np.nan, 2])
    assert metrics["mae"] == 0
    assert counts == {"valid_pixels": 1, "positive_pixels": 1, "nonfinite_predictions": 1}


def test_logarithmic_metrics_clip_nonpositive_predictions():
    metrics, _ = evaluate_height([0], [1])
    assert metrics["mae"] == 1
    assert metrics["rmse_log"] == pytest.approx(-math.log(1e-6))
    assert metrics["log10"] == pytest.approx(6)
    assert metrics["abs_rel"] == pytest.approx(1 - 1e-6)


def test_images_without_positive_heights_have_no_relative_metrics():
    metrics, _ = evaluate_height([1, 2], [0, 0])
    assert set(metrics) == {"mae", "rmse", "bias"}


def test_images_are_averaged_equally_and_metric_counts_are_separate():
    average = HeightMetricAverage()
    average.update(*evaluate_height([2], [1]))
    average.update(*evaluate_height(np.full(100, 3), np.zeros(100)))
    report = average.report()
    assert report["metrics"]["mae"] == 2
    assert report["metrics"]["abs_rel"] == 1
    assert report["metric_image_counts"]["mae"] == 2
    assert report["metric_image_counts"]["abs_rel"] == 1
    assert report["valid_pixels"] == 101


def test_empty_valid_region_and_wrong_shapes_fail():
    with pytest.raises(ValueError, match="no valid pixels"):
        evaluate_height([1], [np.nan])
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_height([1], [1, 2])


def write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "city", "fold"])
        writer.writerows(rows)


def test_validation_selection_is_exact_and_duplicates_fail(tmp_path):
    split = tmp_path / "folds.csv"
    write_split(split, [("one", "ARG", 5), ("two", "ATL", 6), ("three", "JAX", 5)])
    assert runner.load_validation_names(split, 5) == ["ARG_one", "JAX_three"]
    write_split(split, [("one", "ARG", 5), ("one", "ARG", 6)])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_validation_names(split, 5)


def test_prediction_directory_end_to_end_units_and_no_extra_images(tmp_path, monkeypatch):
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    truth.mkdir()
    predictions.mkdir()
    for folder in (truth, predictions):
        for name in ("ARG_one", "ATL_two"):
            (folder / f"{name}_AGL.tif").touch()
    split = tmp_path / "folds.csv"
    write_split(split, [("one", "ARG", 5), ("two", "ATL", 6)])
    calls = []

    def fake_read(path):
        calls.append(path)
        assert path.name == "ARG_one_AGL.tif"
        if path.parent == truth:
            return np.array([[0, 100, 65535]], dtype=np.float32), None
        return np.array([[-3, 2, 100]], dtype=np.float32), None

    output = tmp_path / "report.json"
    monkeypatch.setattr(runner, "read_raster", fake_read)
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda items, **kwargs: items))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_height.py", "--predictions-dir", str(predictions),
        "--truth-dir", str(truth), "--truth-unit", "cm",
        "--folds-csv", str(split), "--fold", "5", "--output", str(output),
    ])
    runner.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["images"] == 1
    assert report["metrics"]["mae"] == 0.5
    assert report["metrics"]["abs_rel"] == 1
    assert report["valid_pixels"] == 2
    assert len(calls) == 2


def test_missing_prediction_is_not_silently_skipped(tmp_path, monkeypatch):
    split = tmp_path / "folds.csv"
    write_split(split, [("one", "ARG", 5)])
    monkeypatch.setattr(sys, "argv", [
        "evaluate_height.py", "--predictions-dir", str(tmp_path),
        "--truth-dir", str(tmp_path), "--folds-csv", str(split),
    ])
    with pytest.raises(FileNotFoundError, match="Missing"):
        runner.main()
