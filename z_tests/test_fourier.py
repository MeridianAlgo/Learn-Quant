import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Fourier Analysis"))
from fourier import (
    autocorrelation,
    detrend,
    dominant_period,
    lowpass,
    power_spectrum,
    reconstruct,
)


def _sine(period, n=256, amplitude=1.0, phase=0.0):
    t = np.arange(n)
    return (amplitude * np.sin(2 * np.pi * t / period + phase)).tolist()


def test_detrend_removes_a_pure_line():
    line = [3.0 + 2.0 * i for i in range(50)]
    assert np.allclose(detrend(line), 0.0, atol=1e-9)


def test_detrend_removes_only_the_line_from_a_trending_wave():
    wave = _sine(16, n=128)
    trended = [v + 0.5 * i + 20.0 for i, v in enumerate(wave)]
    assert np.allclose(detrend(trended), detrend(wave), atol=1e-9)


def test_detrend_needs_two_points():
    with pytest.raises(ValueError):
        detrend([1.0])


def test_spectrum_lengths_match():
    freqs, power = power_spectrum(_sine(20))
    assert len(freqs) == len(power) == 256 // 2 + 1


def test_spectrum_power_is_never_negative():
    _, power = power_spectrum(_sine(20, amplitude=3.0))
    assert min(power) >= 0.0


def test_zero_frequency_holds_the_mean():
    series = [5.0] * 64
    freqs, power = power_spectrum(series)
    assert freqs[0] == 0.0
    # Power at zero frequency is n times the squared mean.
    assert abs(power[0] - 64 * 25.0) < 1e-6


def test_spectrum_rejects_bad_spacing():
    with pytest.raises(ValueError):
        power_spectrum(_sine(20), sample_spacing=0.0)


def test_dominant_period_finds_a_clean_cycle():
    assert abs(dominant_period(_sine(32, n=256)) - 32.0) < 1e-6


def test_dominant_period_finds_the_louder_of_two_cycles():
    mixed = (np.asarray(_sine(40, n=320, amplitude=5.0)) + np.asarray(_sine(10, n=320, amplitude=1.0))).tolist()
    assert abs(dominant_period(mixed) - 40.0) < 1.0


def test_dominant_period_scales_with_sample_spacing():
    # Sampling every half day makes a 32 sample cycle a 16 day one.
    assert abs(dominant_period(_sine(32, n=256), sample_spacing=0.5) - 16.0) < 1e-6


def test_dominant_period_survives_noise():
    rng = np.random.default_rng(0)
    noisy = (np.asarray(_sine(25, n=500, amplitude=4.0)) + rng.normal(0, 1.0, 500)).tolist()
    assert abs(dominant_period(noisy) - 25.0) < 1.5


def test_trend_dominates_until_it_is_removed():
    t = np.arange(400)
    trended = (3.0 * np.sin(2 * np.pi * t / 20) + 0.2 * t).tolist()
    assert dominant_period(trended) > 100
    assert abs(dominant_period(detrend(trended)) - 20.0) < 1.0


def test_min_period_respects_nyquist():
    with pytest.raises(ValueError):
        dominant_period(_sine(20, n=64), min_period=1000.0)


def test_lowpass_keeps_a_slow_wave():
    slow = _sine(50, n=200)
    filtered = lowpass(slow, cutoff_period=20)
    assert np.allclose(filtered, slow, atol=1e-9)


def test_lowpass_deletes_a_fast_wave():
    fast = _sine(5, n=200)
    assert np.allclose(lowpass(fast, cutoff_period=20), 0.0, atol=1e-9)


def test_lowpass_separates_a_mixture():
    slow = np.asarray(_sine(50, n=200, amplitude=2.0))
    fast = np.asarray(_sine(5, n=200, amplitude=1.0))
    filtered = np.asarray(lowpass((slow + fast).tolist(), cutoff_period=20))
    assert np.allclose(filtered, slow, atol=1e-9)


def test_lowpass_never_adds_variance():
    rng = np.random.default_rng(1)
    noisy = rng.normal(0, 1.0, 256).tolist()
    assert np.std(lowpass(noisy, cutoff_period=10)) <= np.std(noisy)


def test_lowpass_rejects_a_bad_cutoff():
    with pytest.raises(ValueError):
        lowpass(_sine(20), cutoff_period=0)


def test_reconstruct_with_everything_returns_the_original():
    rng = np.random.default_rng(2)
    series = rng.normal(0, 1.0, 128).tolist()
    assert np.allclose(reconstruct(series, 128), series, atol=1e-9)


def test_reconstruct_with_one_component_recovers_a_single_sine():
    wave = _sine(16, n=128, amplitude=2.0)
    assert np.allclose(reconstruct(wave, 1), wave, atol=1e-9)


def test_reconstruct_error_falls_as_components_are_added():
    rng = np.random.default_rng(3)
    series = (np.asarray(_sine(20, n=256, amplitude=3.0)) + rng.normal(0, 0.5, 256)).tolist()
    errors = [float(np.max(np.abs(np.asarray(reconstruct(series, k)) - np.asarray(series)))) for k in (1, 5, 20, 60)]
    assert errors == sorted(errors, reverse=True)


def test_reconstruct_keeps_the_mean():
    series = [4.0 + s for s in _sine(16, n=128)]
    assert abs(np.mean(reconstruct(series, 0)) - 4.0) < 1e-9


def test_reconstruct_rejects_negative_components():
    with pytest.raises(ValueError):
        reconstruct(_sine(20), -1)


def test_autocorrelation_starts_at_one():
    rng = np.random.default_rng(4)
    acf = autocorrelation(rng.normal(0, 1.0, 200).tolist())
    assert abs(acf[0] - 1.0) < 1e-12


def test_autocorrelation_of_a_cycle_peaks_at_its_period():
    acf = autocorrelation(_sine(20, n=400), max_lag=45)
    assert acf[20] > 0.8
    assert acf[10] < -0.5


def test_autocorrelation_of_noise_decays_fast():
    rng = np.random.default_rng(5)
    acf = autocorrelation(rng.normal(0, 1.0, 1000).tolist(), max_lag=20)
    assert max(abs(v) for v in acf[1:]) < 0.15


def test_autocorrelation_matches_the_direct_computation():
    rng = np.random.default_rng(6)
    series = rng.normal(0, 1.0, 128)
    centered = series - series.mean()
    acf = autocorrelation(series.tolist(), max_lag=5)
    for lag in range(6):
        direct = float(np.sum(centered[lag:] * centered[: len(centered) - lag]) / np.sum(centered**2))
        assert abs(acf[lag] - direct) < 1e-9


def test_autocorrelation_respects_max_lag():
    assert len(autocorrelation(_sine(20, n=100), max_lag=7)) == 8


def test_autocorrelation_rejects_an_impossible_lag():
    with pytest.raises(ValueError):
        autocorrelation(_sine(20, n=50), max_lag=50)


def test_autocorrelation_rejects_a_flat_series():
    with pytest.raises(ValueError):
        autocorrelation([2.0] * 40)
