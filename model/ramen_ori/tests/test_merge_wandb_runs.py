"""merge_wandb_runs.parse_log / merge_records unit tests (Issue #122)。

wandb 呼出は含まず、log parser と merge rule だけを検証。
"""
from __future__ import annotations

from pathlib import Path

from model.ramen_ori.scripts.merge_wandb_runs import merge_records, parse_log


def test_parse_log_step_line(tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    log.write_text(
        "[step   100] loss=0.5432 lr=1.00e-04 it/s=1.75 elapsed=57.1s\n"
        "[step   200] loss=0.4321 lr=9.99e-05 it/s=1.76 elapsed=113.5s\n"
    )
    records = parse_log(log)
    assert records[100] == {
        "train/loss": 0.5432,
        "train/lr": 1.00e-04,
        "train/it_per_sec": 1.75,
    }
    assert records[200]["train/loss"] == 0.4321
    assert records[200]["train/lr"] == 9.99e-05


def test_parse_log_val_line(tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    log.write_text(
        "[val] step=1000 val/loss_ema=0.1234 (over 20 batches)\n"
        "[val] step=2000 val/loss=0.4567 (over 20 batches)\n"
    )
    records = parse_log(log)
    assert records[1000] == {"val/loss_ema": 0.1234}
    assert records[2000] == {"val/loss": 0.4567}


def test_parse_log_merge_train_and_val_at_same_step(tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    log.write_text(
        "[step  1000] loss=0.3210 lr=5.00e-05 it/s=1.80 elapsed=555.5s\n"
        "[val] step=1000 val/loss_ema=0.2109 (over 20 batches)\n"
    )
    records = parse_log(log)
    assert records[1000] == {
        "train/loss": 0.3210,
        "train/lr": 5.00e-05,
        "train/it_per_sec": 1.80,
        "val/loss_ema": 0.2109,
    }


def test_parse_log_ignores_noise_lines(tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    log.write_text(
        "[data] train source size: 157125, batch_size: 32\n"
        "[step   100] loss=0.5 lr=1e-4 it/s=1.7 elapsed=57s\n"
        "wandb: WARNING Tried to log to step 95000\n"
        "[hf_autopush] pushed ckpt.pt\n"
        "Traceback (most recent call last):\n"
    )
    records = parse_log(log)
    assert list(records) == [100]


def test_merge_records_source_prefer_before_start_step() -> None:
    source = {
        100: {"train/loss": 1.0, "train/lr": 1e-4},
        200: {"train/loss": 0.5, "train/lr": 9e-5},
        300: {"train/loss": 0.3, "train/lr": 8e-5},  # start_step 直後、abandoned
    }
    resume = {
        200: {"train/loss": 999.0, "train/lr": 999.0},  # resume artifact @ start_step
        300: {"train/loss": 0.4, "train/lr": 8e-5},
        400: {"train/loss": 0.2, "train/lr": 7e-5},
    }
    merged = merge_records(source, resume, start_step=200)
    # source: step 100, 200 (step <= 200)
    assert merged[100] == {"train/loss": 1.0, "train/lr": 1e-4}
    assert merged[200] == {"train/loss": 0.5, "train/lr": 9e-5}  # source wins
    # resume: step 300, 400 (step > 200)
    assert merged[300] == {"train/loss": 0.4, "train/lr": 8e-5}  # resume wins
    assert merged[400] == {"train/loss": 0.2, "train/lr": 7e-5}
    assert sorted(merged) == [100, 200, 300, 400]


def test_merge_records_source_only_after_start_step_is_dropped() -> None:
    # source has step 300 (crashed after start_step 200), resume has nothing at 300
    source = {200: {"train/loss": 0.5}, 300: {"train/loss": 0.3}}
    resume = {400: {"train/loss": 0.1}}
    merged = merge_records(source, resume, start_step=200)
    # source step 300 is "abandoned learning" (didn't feed ckpt), dropped
    assert 300 not in merged
    assert sorted(merged) == [200, 400]


def test_merge_records_empty_inputs() -> None:
    assert merge_records({}, {}, start_step=0) == {}


def test_parse_log_lr_with_negative_exponent(tmp_path: Path) -> None:
    # 実 log の様式: lr=1.06e-05 (負 exponent)、resume 中 log でも同じ format
    log = tmp_path / "out.log"
    log.write_text("[step 95000] loss=0.0133 lr=1.06e-05 it/s=13572.93 elapsed=7.0s\n")
    records = parse_log(log)
    assert records[95000]["train/lr"] == 1.06e-05
