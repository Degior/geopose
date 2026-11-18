import argparse
import ast
import io
import os
from pathlib import Path
import sys
import types

import pytest

from training.logging_utils import TrainingProgress, print_main, terminal_progress


ROOT = Path(__file__).resolve().parents[1]


def test_eight_gpu_fifteen_epoch_log_is_bounded_and_rank_zero_only():
    output = io.StringIO()
    metrics = {"loss": 0.7, "huber_agl": 0.8}
    for rank in range(8):
        for epoch in range(15):
            progress = TrainingProgress(epoch, 15, 667, rank=rank, every=100, stream=output)
            for step in range(1, 668):
                progress.update(step, metrics, 3e-6)
            progress.finish(metrics)
    lines = output.getvalue().splitlines()
    assert len(lines) == 15 * 9
    assert sum('step 667/667' in line for line in lines) == 15
    assert sum('finished' in line for line in lines) == 15
    assert '\r' not in output.getvalue()


def test_startup_messages_are_printed_only_by_rank_zero(monkeypatch, capsys):
    monkeypatch.setenv('RANK', '7')
    print_main('not printed')
    monkeypatch.setenv('RANK', '0')
    print_main('printed once')
    assert capsys.readouterr().out == 'printed once\n'


@pytest.mark.parametrize('rank,tty,disabled', [(0, False, True), (7, True, True), (0, True, False)])
def test_progress_bars_only_exist_on_primary_interactive_terminal(monkeypatch, rank, tty, disabled):
    options = {}

    def fake_tqdm(iterable, **kwargs):
        options.update(kwargs)
        return iterable

    monkeypatch.setitem(sys.modules, 'tqdm', types.SimpleNamespace(tqdm=fake_tqdm))
    monkeypatch.setattr(sys, 'stderr', types.SimpleNamespace(isatty=lambda: tty))
    assert list(terminal_progress([1, 2], rank=rank)) == [1, 2]
    assert options['disable'] is disabled


@pytest.mark.parametrize('compact,lines', [(True, 1), (False, 5)])
def test_compact_validation_changes_output_but_preserves_score_and_report(tmp_path, capsys, compact, lines):
    tree = ast.parse((ROOT / 'utilities/evaluate.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_r2')
    scores = {'ARG': 0.8, 'JAX': 0.9, 'OMA': 0.85, 'ATL': 0.95}
    namespace = {'get_city_scores': lambda args, city: scores[city], 'os': os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'evaluate_r2', 'exec'), namespace)
    score = namespace['evaluate_r2'](argparse.Namespace(compact=compact, output_dir=str(tmp_path)))
    assert score == pytest.approx(0.875)
    assert len(capsys.readouterr().out.splitlines()) == lines
    assert 'FINAL R-square: 0.875000' in (tmp_path / 'metrics.txt').read_text()


@pytest.mark.parametrize('batch', [1, 2])
def test_modern_autocast_preserves_scale_head_result_and_gradients(batch):
    torch = pytest.importorskip('torch')
    tree = ast.parse((ROOT / 'utilities/unet_vflow.py').read_text(encoding='utf-8'))
    head = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ScaleHead')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[head], type_ignores=[]), 'ScaleHead', 'exec'), namespace)
    height = torch.arange(1, batch * 4 + 1, dtype=torch.float32).reshape(batch, 1, 2, 2).requires_grad_()
    mag = height.detach() * 2
    expected = (height.detach() * mag).flatten(1).sum(1) / (height.detach().square().flatten(1).sum(1) + 0.01)
    result = namespace['ScaleHead']()(mag, height)
    assert torch.allclose(result.reshape(-1), expected)
    result.sum().backward()
    assert torch.isfinite(height.grad).all()
    assert torch.amp.GradScaler('cuda', enabled=False).is_enabled() is False
