import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import run_height_experiment as experiment
from training.crops import choose_crop


class Coordinates:
    def __init__(self, values):
        self.values = iter(values)

    def randint(self, low, high):
        value = next(self.values)
        assert low <= value <= high
        return value


def test_crop_retries_nodata_and_accepts_valid_zero_heights():
    target = np.full((5, 10), np.nan)
    target[:2, 8:] = 0
    assert choose_crop(target, 2, rng=Coordinates([0, 0, 0, 8])) == (0, 8)


def test_exact_size_crop_needs_no_margin():
    assert choose_crop(np.zeros((32, 32)), 32) == (0, 0)


def test_crop_enforces_minimum_valid_pixels_and_rejects_infinity():
    target = np.zeros((32, 32))
    target[:20] = np.inf
    with pytest.raises(ValueError, match="No valid crop"):
        choose_crop(target, 32, min_valid_fraction=0.5)
    with pytest.raises(ValueError, match="Cannot take"):
        choose_crop(target, 33)


def test_crop_falls_back_to_best_attempt_when_preferred_coverage_is_impossible():
    target = np.full((64, 128), np.nan)
    target[:25, :64] = 0
    target[:30, 64:] = 0
    assert choose_crop(target, 64, min_valid_fraction=0.5, attempts=2,
                       rng=Coordinates([0, 0, 0, 64])) == (0, 64)


def test_reported_1792_crop_with_42_percent_coverage_is_usable():
    target = np.full((1792, 1792), np.nan, dtype=np.float32)
    target.flat[:1352268] = 0
    assert choose_crop(target, 1792, min_valid_fraction=0.5) == (0, 0)


def test_no_fallback_to_all_nodata_crop():
    with pytest.raises(ValueError, match="hard minimum 1024"):
        choose_crop(np.full((64, 64), np.nan), 64, min_valid_fraction=0.5)


def make_args(tmp_path, run_id="test-huber", no_huber=False):
    checkpoint = tmp_path / "author_fold5"
    checkpoint.write_bytes(b"checkpoint for launch configuration test")
    data = tmp_path / "data"
    truth = tmp_path / "truth"
    data.mkdir(exist_ok=True)
    truth.mkdir(exist_ok=True)
    split = tmp_path / "folds.csv"
    with split.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "city", "fold"])
        writer.writerows((str(i), "ARG", 0 if i < 64 else 5) for i in range(72))
    return argparse.Namespace(
        checkpoint=checkpoint, data_dir=data, train_dir=truth, folds_csv=split,
        config=experiment.ROOT / "configs/v2l_huber.json", no_huber=no_huber,
        fold=5, gpus=8, workers=8, cpus=None, epochs=None, run_id=run_id,
        runs_dir=tmp_path / "experiments", python=Path(".venv/bin/python"),
        min_valid_fraction=0.5, memory=80000, time="20:00:00", partition="hiperf", node=None,
    )


def test_eight_gpu_launch_has_expected_batch_schedule_and_isolated_outputs(tmp_path):
    args = make_args(tmp_path)
    stage, manifest, command, sbatch = experiment.prepare(args)
    config = json.loads((stage / "config.json").read_text())
    assert manifest["global_batch"] == 8
    assert manifest["cpus"] == 64
    assert manifest["steps_per_epoch"] == 8
    assert config["optimizer"]["schedule"]["params"]["T_max"] == 120
    assert "--gres=gpu:8" in sbatch
    assert not any(arg.startswith("--nodelist") for arg in sbatch)
    for flag, suffix in [("--output-dir", "weights"), ("--preds-dir", "predictions"), ("--logdir", "tensorboard")]:
        assert command[command.index(flag) + 1] == str(stage / suffix)
    assert command[-1] == "--from-zero"
    assert len(manifest["checkpoint_sha256"]) == 64
    assert any(loss["type"] == "HuberAGLLossCalculator" for loss in config["losses"])
    with pytest.raises(FileExistsError):
        experiment.prepare(args)


def test_control_changes_only_huber_loss_and_output_location(tmp_path):
    huber, _, _, _ = experiment.prepare(make_args(tmp_path, "huber"))
    control, _, _, _ = experiment.prepare(make_args(tmp_path, "control", no_huber=True))
    config_huber = json.loads((huber / "config.json").read_text())
    config_control = json.loads((control / "config.json").read_text())
    config_huber["losses"] = [loss for loss in config_huber["losses"] if loss["type"] != "HuberAGLLossCalculator"]
    assert config_huber == config_control


def test_insufficient_cpu_allocation_is_rejected_before_creating_run(tmp_path):
    args = make_args(tmp_path)
    args.cpus = 8
    with pytest.raises(ValueError, match="CPUs"):
        experiment.prepare(args)
    assert not args.runs_dir.exists()


def test_python_venv_symlink_path_is_not_resolved_to_system_python(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    args.python = tmp_path / ".venv" / "bin" / "python"
    original_resolve = Path.resolve
    system_python = tmp_path / "usr" / "bin" / "python3.9"

    def resolve_like_linux_venv(path, *values, **kwargs):
        if path == args.python:
            return system_python
        return original_resolve(path, *values, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_like_linux_venv)
    _, _, command, sbatch = experiment.prepare(args)
    assert command[0] == str(args.python.absolute())
    assert str(system_python) not in sbatch[-1]


def test_environment_probe_uses_venv_path_and_reports_missing_torch(tmp_path, monkeypatch):
    python = tmp_path / "venv-python"
    python.touch()
    commands = []

    def failed_probe(command, **kwargs):
        commands.append(command)
        return argparse.Namespace(returncode=1, stdout="", stderr="No module named 'torch'")

    monkeypatch.setattr(experiment.subprocess, "run", failed_probe)
    with pytest.raises(RuntimeError, match="No module named 'torch'"):
        experiment.check_training_python(python)
    assert commands[0][0] == str(python.absolute())


def test_huber_value_gradient_and_target_mask():
    torch = pytest.importorskip("torch")
    from training.losses import HuberAGLLossCalculator

    prediction = torch.tensor([[[[2.0, 0.5, 99.0, 99.0]]]], requires_grad=True)
    target = torch.tensor([[[0.0, 1.0, float("nan"), 300.0]]])
    loss = HuberAGLLossCalculator(delta=1.0).calculate_loss({"height": prediction}, {"agl": target})
    assert loss.item() == pytest.approx((1.5 + 0.125) / 2)
    loss.backward()
    assert prediction.grad.flatten().tolist() == pytest.approx([0.5, -0.25, 0.0, 0.0])


def test_huber_empty_targets_keep_differentiable_zero():
    torch = pytest.importorskip("torch")
    from training.losses import HuberAGLLossCalculator

    prediction = torch.tensor([[[[1.0]]]], requires_grad=True)
    loss = HuberAGLLossCalculator().calculate_loss({"height": prediction}, {"agl": torch.tensor([[[float("nan")]]])})
    loss.backward()
    assert loss.item() == 0
    assert prediction.grad.item() == 0
