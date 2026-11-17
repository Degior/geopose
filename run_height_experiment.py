"""Prepare an isolated height fine-tuning experiment, optionally submit to Slurm."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent


def git_branch():
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_training_python(python):
    # Keep the venv symlink: resolving it would select the system environment.
    executable = str(python.absolute())
    if not python.is_file():
        raise FileNotFoundError(f"Training Python not found: {python}")
    probe = subprocess.run(
        [executable, "-c", "import sys, torch, torch.distributed.run; "
         "print('Training Python:', sys.executable, '| prefix:', sys.prefix, '| torch:', torch.__version__)"],
        text=True, capture_output=True,
    )
    if probe.returncode:
        raise RuntimeError(f"Training environment check failed for {executable}:\n{probe.stderr}")
    print(probe.stdout.strip(), flush=True)


def prepare(args):
    checkpoint, data, truth, split = (path.resolve() for path in (
        args.checkpoint, args.data_dir, args.train_dir, args.folds_csv,
    ))
    for path in (checkpoint, split, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not data.is_dir() or not truth.is_dir():
        raise FileNotFoundError("Both prepared data and original ground-truth directories must exist")
    cpus = args.cpus or args.gpus * args.workers
    if cpus < args.gpus * args.workers:
        raise ValueError("Allocate at least gpus * workers CPUs")
    with split.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [f"{row['city']}_{row['id']}" for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate examples in split CSV")
    train_count = sum(int(row["fold"]) != args.fold for row in rows)
    val_count = len(rows) - train_count
    if not train_count or not val_count:
        raise ValueError("Split must have both training and validation examples")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.no_huber:
        config["losses"] = [loss for loss in config["losses"] if loss["type"] != "HuberAGLLossCalculator"]
    if args.epochs is not None:
        config["optimizer"]["schedule"]["epochs"] = args.epochs
    epochs = config["optimizer"]["schedule"]["epochs"]
    batch = config["optimizer"]["train_bs"]
    steps = math.ceil(train_count / args.gpus) // batch
    if steps < 1:
        raise ValueError("Training dataset is too small for this batch and GPU count")
    schedule = config["optimizer"]["schedule"]
    if schedule["type"] != "cosine" or schedule["mode"] != "step":
        raise ValueError("Experiment expects cosine scheduling per optimizer step")
    schedule["params"]["T_max"] = steps * epochs
    run_id = args.run_id or ("control-" if args.no_huber else "huber-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run-id must use letters, digits, underscores or hyphens (up to 96 characters)")
    run = args.runs_dir.resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    stage = run / f"fold_{args.fold}" / "height_tune"
    weights = stage / "weights"
    weights.mkdir(parents=True)
    config_path = stage / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [str(args.python.absolute()), "-u", "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.gpus}", str(ROOT / "training/train_segmentor.py"),
               "--distributed", "--world-size", str(args.gpus), "--config", str(config_path),
               "--workers", str(args.workers), "--data-dir", str(data), "--train-dir", str(truth),
               "--folds-csv", str(split), "--fold", str(args.fold), "--prefix", "folds_",
               "--run-id", run_id, "--stage", "height_tune",
               "--crop", "1792", "--multiplier", "1", "--test_every", "2",
               "--min-valid-fraction", str(args.min_valid_fraction),
               "--output-dir", str(weights), "--logdir", str(stage / "tensorboard"),
               "--preds-dir", str(stage / "predictions"), "--resume", str(checkpoint), "--from-zero"]
    sbatch = ["sbatch", "--nodes=1", "--ntasks=1", f"--gres=gpu:{args.gpus}",
              f"--cpus-per-task={cpus}", f"--mem={args.memory}", f"--time={args.time}",
              f"--partition={args.partition}", f"--job-name=geopose-{run_id[:40]}",
              f"--chdir={ROOT}", f"--output={stage / 'slurm_%j.log'}", "--export=ALL"]
    if args.node:
        sbatch.append(f"--nodelist={args.node}")
    sbatch.append("--wrap=" + "export PYTHONPATH=" + shlex.quote(str(ROOT)) + "; exec " + shlex.join(command))
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "branch": git_branch(), "fold": args.fold,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "split": str(split), "split_sha256": sha256(split),
                "train_images": train_count, "validation_images": val_count,
                "epochs": epochs, "gpus": args.gpus, "workers_per_gpu": args.workers,
                "cpus": cpus, "batch_per_gpu": batch, "global_batch": batch * args.gpus,
                "steps_per_epoch": steps, "scheduler_steps": steps * epochs,
                "min_valid_fraction": args.min_valid_fraction,
                "weights": str(weights), "command": command, "sbatch": sbatch}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return stage, manifest, command, sbatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, default=ROOT / "folds.csv")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2l_huber.json")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory", type=int, default=80000)
    parser.add_argument("--partition", default="hiperf")
    parser.add_argument("--node", help="Optional; otherwise Slurm chooses a node with enough GPUs")
    parser.add_argument("--time", default="20:00:00")
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-huber", action="store_true", help="Prepare an otherwise identical control run")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "weights/experiments")
    parser.add_argument("--run-id")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1 or args.workers < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("GPU, worker and epoch counts must be positive")
    if not 0 <= args.min_valid_fraction <= 1:
        parser.error("--min-valid-fraction must be between 0 and 1")
    if args.submit or args.local:
        check_training_python(args.python)
    stage, manifest, command, sbatch = prepare(args)
    print(f"Run: {stage}")
    print(f"GPU: {args.gpus}; global batch: {manifest['global_batch']}; T_max: {manifest['scheduler_steps']}")
    print(shlex.join(sbatch))
    if args.submit:
        try:
            result = subprocess.run(sbatch, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            print(error.stderr, file=sys.stderr)
            raise
        (stage / "submission.txt").write_text(result.stdout, encoding="utf-8")
        print(result.stdout.strip())
    elif args.local:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print("Prepared only. The manifest contains the submission command; use --submit on a new run to submit directly.")


if __name__ == "__main__":
    main()
