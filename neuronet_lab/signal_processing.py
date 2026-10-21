"""Conservative offline pulse metrics; no bridging recording gaps.

The detector is a heuristic, not a clinically validated QRS/PPG algorithm.
PSD uses a one-sided Hann periodogram, with RR expressed in milliseconds.
"""
from __future__ import annotations

import numpy as np


def smooth(values, width):
    width = max(1, min(len(values), int(width)))
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    running = np.cumsum(np.r_[0.0, padded], dtype=np.float64)
    return (running[width:] - running[:-width]) / width


def periodogram(values, fs):
    """Return Hz and density (input units squared / Hz), including one-sided scaling."""
    x = np.asarray(values, dtype=np.float64)
    weights = np.hanning(len(x))
    density = np.abs(np.fft.rfft((x - x.mean()) * weights)) ** 2 / (fs * np.sum(weights ** 2))
    density[1:-1 if len(x) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(x), 1 / fs), density


def _segment_peaks(x, fs, kind):
    centered = x - smooth(x, max(3, fs * 0.55))
    if kind == "ecg":
        # Squared derivative identifies QRS activity irrespective of lead polarity.
        activity = smooth(np.diff(centered, prepend=centered[0]) ** 2, max(2, fs * 0.08))
        threshold = max(np.percentile(activity, 88), np.median(activity) * 2)
        distance = max(1, int(fs * 0.25))
    else:
        activity = smooth(centered, max(2, fs * 0.05))
        threshold = np.percentile(activity, 55)
        distance = max(1, int(fs * 0.30))
    candidates = np.flatnonzero((activity[1:-1] > activity[:-2]) & (activity[1:-1] >= activity[2:])) + 1
    candidates = candidates[activity[candidates] >= threshold]
    selected = []
    for p in candidates:
        if not selected or p - selected[-1] >= distance:
            selected.append(int(p))
        elif activity[p] > activity[selected[-1]]:
            selected[-1] = int(p)
    if kind == "ecg":
        refined = []
        radius = max(1, int(fs * 0.10))
        for p in selected:
            left, right = max(0, p - radius), min(len(x), p + radius + 1)
            r = left + int(np.argmax(np.abs(centered[left:right])))
            if not refined or r - refined[-1] >= distance:
                refined.append(r)
        selected = refined
    return np.asarray(selected, dtype=int)


def pulse_peaks(values, fs, kind, times=None):
    x = np.asarray(values, dtype=np.float64)
    if fs <= 0 or len(x) < fs * 3:
        return np.array([], dtype=int)
    valid = np.isfinite(x)
    breaks = np.zeros(len(x), dtype=bool)
    if times is not None and len(times) == len(x):
        dt = np.diff(times)
        breaks[1:] = (dt > 1.5 / fs) | (dt <= 0)
    starts = np.flatnonzero(valid & (np.r_[True, ~valid[:-1]] | breaks))
    ends = np.flatnonzero(valid & (np.r_[~valid[1:], True] | np.r_[breaks[1:], False])) + 1
    detected = [_segment_peaks(x[a:b], fs, kind) + a for a, b in zip(starts, ends) if b - a >= fs * 3]
    return np.concatenate(detected) if detected else np.array([], dtype=int)


def heart_stats(signal, fs, kind, duration, times=None):
    result = {key: None for key in ("heart_rate_bpm", "sdnn_ms", "rmssd_ms", "ulf_power", "vlf_power", "lf_power", "hf_power", "total_power", "lf_hf", "spectrum")}
    result.update(quality="insufficient_beats", rr_count=0, rejected_rr_count=0,
                  power_unit="ms²", total_power_band_hz=[0.04, 0.4],
                  spectral_limits="ULF/VLF unavailable on 120-second windows")
    x = np.asarray(signal, dtype=np.float64)
    if fs <= 0 or len(x) < 2:
        return result
    clock = np.asarray(times, dtype=np.float64) if times is not None else np.arange(len(x)) / fs
    peaks = pulse_peaks(x, fs, kind, clock)
    if len(peaks) < 3:
        return result
    rr = np.diff(clock[peaks])
    valid = (rr >= 0.3) & (rr <= 2.0)
    bad_edges = ~np.isfinite(x)
    bad_edges[1:] |= (np.diff(clock) > 1.5 / fs) | (np.diff(clock) <= 0)
    cumulative_bad = np.cumsum(bad_edges)
    valid &= cumulative_bad[peaks[1:]] == cumulative_bad[peaks[:-1]]
    # Conservative local outlier screening. Rejected beats keep their original time.
    for i in range(len(rr)):
        neighbors = rr[max(0, i - 3):i + 4]
        neighbors = neighbors[(neighbors >= 0.3) & (neighbors <= 2)]
        if len(neighbors) >= 3 and abs(rr[i] - np.median(neighbors)) > 0.25 * np.median(neighbors):
            valid[i] = False
    result.update(rr_count=int(valid.sum()), rejected_rr_count=int((~valid).sum()))
    if valid.sum() < 2:
        return result
    result["heart_rate_bpm"] = float(60 / np.median(rr[valid]))
    result["quality"] = "insufficient_duration"
    if duration + 1 / fs < 120:
        return result
    observed = float(np.isfinite(x).sum() / fs)
    if valid.sum() < 50 or valid.mean() < 0.95 or observed < duration * 0.95 or np.any(np.diff(clock) > 1.5 / fs):
        result["quality"] = "artifacts_or_gaps"
        return result
    accepted = rr[valid] * 1000
    result["sdnn_ms"] = float(np.std(accepted, ddof=1))
    adjacent = valid[:-1] & valid[1:]
    if adjacent.any():
        result["rmssd_ms"] = float(np.sqrt(np.mean((np.diff(rr)[adjacent] * 1000) ** 2)))
    beat_time = clock[peaks[1:]][valid]
    if beat_time[-1] - beat_time[0] < duration * 0.9 or np.max(np.diff(beat_time)) > 3:
        result["quality"] = "insufficient_contiguous_rr"
        return result
    grid = np.arange(beat_time[0], beat_time[-1], 0.25)
    freq, power = periodogram(np.interp(grid, beat_time, accepted), 4.0)
    df = freq[1] - freq[0]
    lf = float(power[(freq >= 0.04) & (freq < 0.15)].sum() * df)
    hf = float(power[(freq >= 0.15) & (freq <= 0.4)].sum() * df)
    result.update(lf_power=lf, hf_power=hf, total_power=lf + hf,
                  lf_hf=lf / hf if hf > 1e-12 else None, quality="ok")
    display = (freq >= 0.0033) & (freq <= 0.4)
    result["spectrum"] = {"hz": freq[display].tolist(), "power": power[display].tolist(), "unit": "ms²/Hz"}
    return result


def respiration(signal, fs, times=None):
    x = np.asarray(signal, dtype=np.float64)
    if fs <= 0 or len(x) < fs * 20 or not np.isfinite(x).all():
        return None
    if times is not None and np.any(np.abs(np.diff(times) - 1 / fs) > 0.5 / fs):
        return None
    # No decimation: retains the true time grid and avoids aliasing.
    freq, power = periodogram(x, fs)
    band = (freq >= 0.08) & (freq <= 0.6)
    if not band.any() or power[band].max() <= 1e-20:
        return None
    return float(freq[band][np.argmax(power[band])] * 60)
