"""
Fourier Analysis
----------------
A price series is a sum of waves if you are willing to look at it that way.
The Fourier transform does exactly that, taking a series indexed by time and
handing back the amplitude of every frequency hiding inside it. The result is
a different view of the same data, and questions that are awkward in the time
domain become easy in the frequency domain.

Three of those questions show up constantly. Is there a repeating cycle in
this series and how long is it. Can I strip the fast wiggles out without the
lag a moving average would introduce. How quickly does a series forget its
own past. The last one is autocorrelation, which the transform computes for
every lag at once far faster than the direct sum would.

One warning before any of it. A trend is not a cycle, but the transform has
no way to say so and will spend most of the spectrum trying to describe the
trend with low frequency waves. Detrend first.
"""

from __future__ import annotations

import numpy as np


def detrend(series: list[float]) -> list[float]:
    """Return the series with its least squares straight line removed.

    A rising series looks to the transform like the first half of a very slow
    wave, and that fake wave swamps every real cycle in the spectrum. Removing
    the line first is not optional preprocessing, it is what makes the rest of
    the answers mean anything.
    """
    y = np.asarray(series, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two points to fit a line")
    x = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return (y - (slope * x + intercept)).tolist()


def power_spectrum(series: list[float], sample_spacing: float = 1.0) -> tuple[list[float], list[float]]:
    """Return frequencies and their power for a real valued series.

    Power is the squared magnitude of each frequency component, scaled by the
    length of the series so that spectra of different lengths are comparable.
    The zero frequency term is the mean and it is returned along with the rest
    rather than silently dropped, since knowing it is there is part of
    understanding why detrending matters.
    """
    y = np.asarray(series, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two points")
    if sample_spacing <= 0:
        raise ValueError("sample spacing must be positive")
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=sample_spacing)
    power = (np.abs(spectrum) ** 2) / y.size
    return freqs.tolist(), power.tolist()


def dominant_period(
    series: list[float],
    sample_spacing: float = 1.0,
    min_period: float = 2.0,
) -> float:
    """Return the length of the strongest cycle in the series.

    The answer is in the same units as `sample_spacing`, so daily data gives
    an answer in days. Frequencies shorter than `min_period` are ignored
    because nothing below two samples per cycle is real, that is the Nyquist
    limit, and anything faster than the sampling rate comes back disguised as
    something slower.
    """
    freqs, power = power_spectrum(series, sample_spacing)
    f = np.asarray(freqs)
    p = np.asarray(power)
    # Drop the zero frequency mean term and anything faster than Nyquist allows.
    usable = (f > 0) & (1.0 / np.where(f > 0, f, 1.0) >= min_period)
    if not np.any(usable):
        raise ValueError("no usable frequencies, the series is too short")
    best = np.argmax(np.where(usable, p, -np.inf))
    return float(1.0 / f[best])


def lowpass(series: list[float], cutoff_period: float, sample_spacing: float = 1.0) -> list[float]:
    """Return the series with every cycle faster than `cutoff_period` removed.

    This is a smoother with a property no moving average has, which is that it
    introduces no lag. The cost is that it uses the whole series including
    points after the one being smoothed, so it is a research tool for looking
    at history and not something you can run live without leaking the future
    into your signal.
    """
    if cutoff_period <= 0:
        raise ValueError("cutoff period must be positive")
    y = np.asarray(series, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two points")
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=sample_spacing)
    spectrum[freqs > 1.0 / cutoff_period] = 0.0
    return np.fft.irfft(spectrum, n=y.size).tolist()


def reconstruct(series: list[float], n_components: int) -> list[float]:
    """Return the series rebuilt from only its strongest frequencies.

    Keeping the mean plus the few loudest components gives a compressed
    version of the series that captures its shape and discards the rest. Push
    `n_components` high enough and you get the original back exactly, which is
    a useful sanity check and a reminder that the transform loses nothing.
    """
    if n_components < 0:
        raise ValueError("cannot keep a negative number of components")
    y = np.asarray(series, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two points")
    spectrum = np.fft.rfft(y)
    power = np.abs(spectrum) ** 2
    # The mean term is always kept, so rank only the real oscillating parts.
    order = np.argsort(power[1:])[::-1] + 1
    keep = set(order[:n_components].tolist()) | {0}
    filtered = np.array([c if i in keep else 0.0 for i, c in enumerate(spectrum)])
    return np.fft.irfft(filtered, n=y.size).tolist()


def autocorrelation(series: list[float], max_lag: int | None = None) -> list[float]:
    """Return the autocorrelation of the series for lags zero upward.

    Computed through the transform rather than by summing products at every
    lag, which turns an order n squared job into an order n log n one. Lag
    zero is always exactly one because a series is perfectly correlated with
    itself. How fast the values fall away from there tells you how long the
    series remembers.
    """
    y = np.asarray(series, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two points")
    if max_lag is None:
        max_lag = y.size - 1
    if max_lag < 0 or max_lag >= y.size:
        raise ValueError("max lag must sit between zero and one less than the series length")
    centered = y - y.mean()
    if not np.any(centered):
        raise ValueError("a flat series has no autocorrelation")
    # Zero padding to at least twice the length stops the circular wraparound
    # that the transform would otherwise introduce.
    size = int(2 ** np.ceil(np.log2(2 * y.size)))
    spectrum = np.fft.rfft(centered, n=size)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n=size)[: max_lag + 1]
    return (acf / acf[0]).tolist()


if __name__ == "__main__":
    print("Fourier Analysis")
    print("=" * 40)

    rng = np.random.default_rng(11)
    n = 512
    t = np.arange(n)
    # A twenty day cycle, a slow drift and a lot of noise on top.
    signal = 3.0 * np.sin(2 * np.pi * t / 20) + 0.05 * t + rng.normal(0, 1.0, n)

    print("\nA twenty day cycle buried in drift and noise")
    print(f"  dominant period, raw series       {dominant_period(signal.tolist()):.2f}")
    clean = detrend(signal.tolist())
    print(f"  dominant period, after detrending {dominant_period(clean):.2f}")
    print("\nUntouched, the strongest wave found is one that spans the whole sample.")
    print("That is the drift wearing a cycle costume. Detrend first, always.")

    smoothed = lowpass(clean, cutoff_period=10)
    print("\nLow pass filter, cycles faster than ten days removed")
    print(f"  standard deviation before {np.std(clean):.4f}")
    print(f"  standard deviation after  {np.std(smoothed):.4f}")

    print("\nRebuilding from a handful of frequencies")
    for k in (1, 3, 10, 260):
        error = np.max(np.abs(np.asarray(reconstruct(clean, k)) - np.asarray(clean)))
        print(f"  {k:>3} components, worst error {error:.6f}")

    acf = autocorrelation(clean, max_lag=40)
    print("\nAutocorrelation of the detrended series")
    for lag in (0, 5, 10, 20, 40):
        print(f"  lag {lag:>2} {acf[lag]:+.4f}")
    print("\nPeaks at twenty and forty, troughs at ten and thirty. That is the cycle.")
