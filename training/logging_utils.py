"""Bounded text output for batch jobs; interactive progress only on terminals."""

import os
import sys
import time


def print_main(*values, rank=None, **kwargs):
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print(*values, flush=True, **kwargs)


def terminal_progress(iterable, rank=0, **kwargs):
    from tqdm import tqdm

    return tqdm(iterable, disable=rank != 0 or not sys.stderr.isatty(),
                mininterval=2.0, ascii=True, **kwargs)


class TrainingProgress:
    def __init__(self, epoch, epochs, steps, rank=0, every=100, stream=None):
        if every < 1:
            raise ValueError("Logging interval must be positive")
        self.epoch, self.epochs, self.steps = epoch, epochs, steps
        self.rank, self.every = rank, every
        self.stream = stream if stream is not None else sys.stdout
        self.started = time.monotonic()

    def update(self, step, metrics, lr):
        if self.rank != 0 or (step != 1 and step != self.steps and step % self.every):
            return
        selected = {name: metrics[name] for name in ("loss", "huber_agl") if name in metrics}
        values = " ".join(f"{name}={value:.5f}" for name, value in selected.items())
        print(f"Epoch {self.epoch + 1}/{self.epochs} | step {step}/{self.steps} | "
              f"lr={lr:.3g} | {values} | elapsed={time.monotonic() - self.started:.0f}s",
              file=self.stream, flush=True)

    def finish(self, metrics):
        if self.rank == 0:
            values = " ".join(f"{name}={value:.5f}" for name, value in metrics.items())
            print(f"Epoch {self.epoch + 1}/{self.epochs} finished (rank 0 averages) | {values}",
                  file=self.stream, flush=True)
