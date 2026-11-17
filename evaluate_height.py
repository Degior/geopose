"""Evaluate a GeoPose checkpoint's AGL output using depth/height metrics."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from utilities.height_metrics import HeightMetricAverage, evaluate_height


ROOT = Path(__file__).resolve().parent


def load_validation_names(path, fold):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "city", "fold"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV must contain id, city, fold columns")
        rows = list(reader)
    all_names = [f"{row['city']}_{row['id']}" for row in rows]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Duplicate image IDs in split CSV")
    names = []
    for row, name in zip(rows, all_names):
        if row["city"] not in {"ARG", "ATL", "JAX", "OMA"}:
            raise ValueError(f"Unsupported city: {row['city']}")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid image ID: {name}")
        if int(row["fold"]) == fold:
            names.append(name)
    if not names:
        raise ValueError(f"No validation images for fold {fold}")
    return names


def read_raster(path):
    from osgeo import gdal

    dataset = gdal.Open(str(path))
    if dataset is None:
        raise OSError(f"Cannot open raster: {path}")
    values = dataset.ReadAsArray()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    if values is None:
        raise OSError(f"Cannot read raster: {path}")
    return values, nodata


def load_target(path, unit, nan_placeholder):
    values, nodata = read_raster(path)
    target = values.astype(np.float32)
    if target.ndim != 2:
        raise ValueError(f"Expected a single-channel AGL raster: {path}")
    invalid = ~np.isfinite(target) | (target == nan_placeholder)
    if nodata is not None:
        invalid |= target == nodata
    target[invalid] = np.nan
    return target / {"m": 1.0, "cm": 100.0}[unit]


def load_model(args, device):
    import torch
    from utilities import unet

    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    model_class = getattr(unet, config["network"])
    # All parameters come from the checkpoint; no ImageNet download is needed.
    model = model_class(**config["encoder_params"], pretrained=False)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    return model, checkpoint.get("epoch")


def predict_height(model, rgb_path, target_shape, device):
    import torch
    import torch.nn.functional as F

    rgb, _ = read_raster(rgb_path)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected a three-channel RGB raster: {rgb_path}; got {rgb.shape}")
    height, width = rgb.shape[1:]
    rgb = rgb.astype(np.float32) / 255.0
    image = torch.from_numpy(np.ascontiguousarray((rgb - 0.5) * 2.0)).unsqueeze(0).to(device)
    image = F.pad(image, (0, -width % 32, 0, -height % 32), mode="replicate")
    with torch.inference_mode():
        prediction = model(image)["height"][:, :, :height, :width]
        # Match Depth Anything's validation resizing convention when necessary.
        if prediction.shape[-2:] != target_shape:
            prediction = F.interpolate(prediction, target_shape, mode="bilinear", align_corners=True)
    return prediction[0, 0].float().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-path", type=Path, help="One checkpoint, matching the evaluated fold")
    mode.add_argument("--predictions-dir", type=Path, help="Existing NAME_AGL.tif predictions in metres")
    parser.add_argument("--data-dir", type=Path, help="Prepared RGB TIFF directory; required for model inference")
    parser.add_argument("--truth-dir", required=True, type=Path, help="Ground-truth AGL TIFF directory")
    parser.add_argument("--truth-unit", choices=("m", "cm"), default="cm")
    parser.add_argument("--folds-csv", type=Path, default=ROOT / "folds.csv")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2l.json")
    parser.add_argument("--max-height", type=float, default=250.0, help="Same upper GT bound as Depth Anything, in metres")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--nan-placeholder", type=float, default=65535)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/height_fold5.json")
    args = parser.parse_args()
    if args.model_path is not None and args.data_dir is None:
        parser.error("--data-dir is required with --model-path")
    if not np.isfinite(args.max_height) or args.max_height <= 0:
        parser.error("--max-height must be positive and finite")
    if not np.isfinite(args.eps) or args.eps <= 0:
        parser.error("--eps must be positive and finite")
    return args


def main():
    args = parse_args()
    names = load_validation_names(args.folds_csv, args.fold)
    needed = [args.truth_dir / f"{name}_AGL.tif" for name in names]
    if args.model_path is not None:
        needed += [args.model_path, args.config]
        needed += [args.data_dir / f"{name}_RGB.tif" for name in names]
    else:
        needed += [args.predictions_dir / f"{name}_AGL.tif" for name in names]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required files, first: {missing[:5]}")

    model, checkpoint_epoch, device = None, None, None
    if args.model_path is not None:
        import torch

        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check GPU allocation or use --device cpu")
        model, checkpoint_epoch = load_model(args, device)
    from tqdm import tqdm

    average = HeightMetricAverage()
    for name in tqdm(names, desc=f"AGL metrics, fold {args.fold}"):
        target = load_target(args.truth_dir / f"{name}_AGL.tif", args.truth_unit, args.nan_placeholder)
        if model is not None:
            prediction = predict_height(model, args.data_dir / f"{name}_RGB.tif", target.shape, device)
        else:
            prediction, _ = read_raster(args.predictions_dir / f"{name}_AGL.tif")
        # GeoPose's validation also clips negative predicted heights to zero.
        prediction = np.maximum(prediction, 0.0)
        try:
            metrics, counts = evaluate_height(prediction, target, args.max_height, args.eps)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        average.update(metrics, counts)
    report = average.report()
    report["settings"] = {
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "checkpoint_epoch": checkpoint_epoch,
        "predictions_dir": str(args.predictions_dir.resolve()) if args.predictions_dir else None,
        "data_dir": str(args.data_dir.resolve()) if args.data_dir else None,
        "truth_dir": str(args.truth_dir.resolve()),
        "truth_unit": args.truth_unit,
        "folds_csv": str(args.folds_csv.resolve()),
        "fold": args.fold,
        "max_height_m": args.max_height,
        "eps_m": args.eps,
        "averaging": "per image, then arithmetic mean per metric",
        "absolute_mask": "finite prediction and GT, 0 <= GT <= max_height",
        "relative_mask": "absolute_mask and GT > eps; prediction >= eps",
        "negative_predictions": "clipped to zero before evaluation, as in GeoPose validation",
        "silog": "sqrt(mean(log_error^2) - 0.5 * mean(log_error)^2), without x100",
        "inference": "full image, FP32, no TTA; bilinear resize with align_corners=True if needed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["metrics"], indent=2, allow_nan=False))
    print(f"Evaluated {report['images']} images; report: {args.output}")
    if report["nonfinite_predictions"]:
        print(f"Excluded {report['nonfinite_predictions']} non-finite predicted pixels; see report")


if __name__ == "__main__":
    main()
