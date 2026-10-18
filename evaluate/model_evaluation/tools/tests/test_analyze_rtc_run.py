"""`analyze_rtc_run` の純粋計算部分の test (Issue #137)。

wandb / URDF に触らない部分だけを対象にする。帯振幅の定義は本 script 固有で、
run 間の相対比較にしか使えない値なので、**定義が変わっていないこと**を
機械的に固定しておくのが目的。
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate.model_evaluation.tools.analyze_rtc_run import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    band_amplitude_mm,
)


def _signal(freq_hz: float, amplitude_mm: float, *, hz: float = 30.0, seconds: float = 10.0):
    times = np.arange(0.0, seconds, 1.0 / hz)
    positions = np.zeros((times.size, 3))
    # x 軸だけに正弦を入れる。単位は m なので mm → m に直す
    positions[:, 0] = (amplitude_mm / 1000.0) * np.sin(2 * np.pi * freq_hz * times)
    return positions, times


def test_recovers_a_known_in_band_sine() -> None:
    positions, times = _signal(2.0, 5.0)
    amplitude, peak_hz = band_amplitude_mm(positions, times)
    assert amplitude == pytest.approx(5.0, rel=0.05)
    assert peak_hz == pytest.approx(2.0, abs=0.15)


def test_out_of_band_content_is_rejected() -> None:
    # 0.5Hz は帯域外。通常の動作 (teacher 0.80-0.85Hz) を拾わないことの確認
    positions, times = _signal(0.5, 20.0)
    amplitude, _peak = band_amplitude_mm(positions, times)
    assert amplitude < 1.0


def test_band_edges() -> None:
    inside, times = _signal((BAND_LOW_HZ + BAND_HIGH_HZ) / 2.0, 4.0)
    outside, _ = _signal(BAND_HIGH_HZ * 2.0, 4.0)
    assert band_amplitude_mm(inside, times)[0] > band_amplitude_mm(outside, times)[0]


def test_axes_are_combined_in_quadrature() -> None:
    positions, times = _signal(2.0, 3.0)
    positions[:, 1] = positions[:, 0]  # 同じ成分を y にも入れる
    amplitude, _peak = band_amplitude_mm(positions, times)
    assert amplitude == pytest.approx(3.0 * np.sqrt(2.0), rel=0.05)


def test_constant_signal_has_no_band_content() -> None:
    times = np.arange(0.0, 10.0, 1.0 / 30.0)
    positions = np.full((times.size, 3), 0.42)
    amplitude, _peak = band_amplitude_mm(positions, times)
    assert amplitude == pytest.approx(0.0, abs=1e-6)


def test_irregular_sampling_is_resampled() -> None:
    # policy_time_s は僅かに不均一。ジッタを入れても振幅が大きく崩れないこと
    positions, times = _signal(2.0, 5.0)
    rng = np.random.default_rng(0)
    jittered = times + rng.normal(0.0, 0.001, size=times.size)
    jittered.sort()
    amplitude, peak_hz = band_amplitude_mm(positions, jittered)
    assert amplitude == pytest.approx(5.0, rel=0.10)
    assert peak_hz == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("rows", [0, 1, 7])
def test_too_few_samples_returns_nan(rows: int) -> None:
    positions = np.zeros((rows, 3))
    times = np.arange(rows, dtype=float) / 30.0
    amplitude, peak = band_amplitude_mm(positions, times)
    assert np.isnan(amplitude) and np.isnan(peak)
