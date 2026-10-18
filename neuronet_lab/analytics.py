"""Interval metrics. Values without enough signal are explicitly unavailable."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .bioradio import window


def _finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(values):
    values = _finite(values)
    return round(float(np.mean(values)), 2) if len(values) else None


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, width)
    if width == 1:
        return values
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    running = np.cumsum(np.r_[0.0, padded], dtype=np.float64)
    return (running[width:] - running[:-width]) / width


def _peaks(values: np.ndarray, fs: float, kind: str) -> np.ndarray:
    if fs <= 0 or len(values) < fs * 3:
        return np.array([], dtype=int)
    x = np.asarray(values, dtype=np.float64)
    good = np.isfinite(x)
    if good.sum() < fs * 3:
        return np.array([], dtype=int)
    x = np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])
    if kind == "ecg":
        baseline = _smooth(x, max(3, int(fs * 0.55)))
        activity = _smooth(np.diff(x - baseline, prepend=x[0]) ** 2, max(2, int(fs * 0.08)))
        threshold = max(np.percentile(activity, 88), np.median(activity) * 2)
        distance = max(1, int(fs * 0.35))
    else:
        width = max(2, int(fs * 0.05))
        activity = _smooth(x, width)
        threshold = np.percentile(activity, 55)
        distance = max(1, int(fs * 0.4))
    candidates = np.flatnonzero((activity[1:-1] > activity[:-2]) & (activity[1:-1] >= activity[2:])) + 1
    candidates = candidates[activity[candidates] >= threshold]
    selected = []
    for peak in candidates:
        if not selected or peak - selected[-1] >= distance:
            selected.append(int(peak))
        elif activity[peak] > activity[selected[-1]]:
            selected[-1] = int(peak)
    return np.asarray(selected, dtype=int)


def _heart_stats(signal, fs, kind, duration):
    unavailable = {"heart_rate_bpm": None, "sdnn_ms": None, "rmssd_ms": None, "ulf_power": None, "vlf_power": None, "lf_power": None, "hf_power": None, "total_power": None, "lf_hf": None, "spectrum": None}
    peaks = _peaks(signal, fs, kind)
    if len(peaks) < 3:
        return unavailable
    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    if len(intervals) < 2:
        return unavailable
    bpm = round(60 / float(np.median(intervals)), 1)
    result = dict(unavailable, heart_rate_bpm=bpm)
    if duration < 120 or len(intervals) < 50:
        return result
    result["sdnn_ms"] = round(float(np.std(intervals, ddof=1) * 1000), 1)
    result["rmssd_ms"] = round(float(np.sqrt(np.mean(np.diff(intervals) ** 2)) * 1000), 1)
    beat_time = np.cumsum(intervals)
    grid = np.arange(beat_time[0], beat_time[-1], 0.25)
    if len(grid) < 100:
        return result
    tachogram = np.interp(grid, beat_time, intervals)
    tachogram = tachogram - np.mean(tachogram)
    spectrum = np.abs(np.fft.rfft(tachogram * np.hanning(len(tachogram)))) ** 2
    freq = np.fft.rfftfreq(len(tachogram), d=0.25)
    step_hz = freq[1] - freq[0]
    lf = np.sum(spectrum[(freq >= 0.04) & (freq < 0.15)]) * step_hz
    hf = np.sum(spectrum[(freq >= 0.15) & (freq <= 0.4)]) * step_hz
    vlf = np.sum(spectrum[(freq >= 0.0033) & (freq < 0.04)]) * step_hz
    result["lf_hf"] = round(float(lf / hf), 3) if hf > 0 else None
    result["vlf_power"] = round(float(vlf), 4)
    result["lf_power"] = round(float(lf), 4)
    result["hf_power"] = round(float(hf), 4)
    result["total_power"] = round(float(np.sum(spectrum[(freq >= 0.0033) & (freq <= 0.4)]) * step_hz), 4)
    display = (freq >= 0.0033) & (freq <= 0.4)
    result["spectrum"] = {"hz": np.round(freq[display], 4).tolist(), "power": np.round(spectrum[display], 5).tolist()}
    return result


def _respiration(signal, fs):
    x = _finite(signal)
    if fs <= 0 or len(x) < fs * 20:
        return None
    x = x[:: max(1, int(fs / 10))]
    fs_down = fs / max(1, int(fs / 10))
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), d=1 / fs_down)
    mask = (freq >= 0.08) & (freq <= 0.6)
    if not mask.any() or not np.max(power[mask]):
        return None
    return round(float(freq[mask][np.argmax(power[mask])] * 60), 1)


def _speech_stats(turns: list[dict], start: float, end: float) -> dict:
    selected = [t for t in turns if t.get("end", 0) > start and t.get("start", 0) < end]
    respondent = [t for t in selected if t.get("speaker") == "respondent"]
    words = sum(len(re.findall(r"[\wа-яё]+", t.get("text", ""), re.I)) for t in respondent)
    syllables = sum(len(re.findall(r"[аеёиоуыэюяaeiouy]", t.get("text", ""), re.I)) for t in respondent)
    speaking_seconds = sum(max(0, min(end, t["end"]) - max(start, t["start"])) for t in respondent)
    latencies = []
    ordered = sorted(selected, key=lambda t: t.get("start", 0))
    for i, turn in enumerate(ordered[:-1]):
        if turn.get("speaker") == "interviewer":
            following = next((x for x in ordered[i + 1 :] if x.get("speaker") == "respondent"), None)
            if following:
                latencies.append(max(0, following["start"] - turn["end"]))
    return {
        "respondent_words": words if respondent else None,
        "words_per_minute": round(words * 60 / speaking_seconds, 1) if speaking_seconds else None,
        "syllables_per_second": round(syllables / speaking_seconds, 2) if speaking_seconds else None,
        "response_latency_seconds": round(float(np.mean(latencies)), 2) if latencies else None,
        "turn_count": len(selected),
        "response_duration_seconds": _mean([t["end"] - t["start"] for t in respondent]),
        "internal_pause_seconds": _mean([t.get("pause_seconds") for t in respondent]),
        "voice_rms": _mean([t.get("acoustic", {}).get("rms") for t in respondent]),
        "pitch_hz": _mean([t.get("acoustic", {}).get("pitch_hz") for t in respondent]),
        "spectral_centroid_hz": _mean([t.get("acoustic", {}).get("spectral_centroid_hz") for t in respondent]),
    }


def _video_stats(session_dir: Path, start: float, end: float) -> dict:
    source = session_dir / "video_features.json"
    if not source.exists():
        return {"rppg_heart_rate_bpm": None, "head_motion": None, "hand_motion": None, "leg_motion": None, "trunk_tilt_degrees": None, "quality": "not_processed"}
    points = json.loads(source.read_text(encoding="utf-8"))
    selection = [p for p in points if start <= p["time"] <= end]
    valid_bpm = [p["rppg_bpm"] for p in selection if p.get("rppg_bpm") is not None]
    return {
        "rppg_heart_rate_bpm": _mean(valid_bpm),
        "head_motion": _mean([p.get("head_motion") for p in selection]),
        "hand_motion": _mean([p.get("hand_motion") for p in selection]),
        "leg_motion": _mean([p.get("leg_motion") for p in selection]),
        "trunk_tilt_degrees": _mean([p.get("trunk_tilt") for p in selection]),
        "quality": "ok" if valid_bpm else "face_signal_unavailable",
    }


def interval_stats(session_dir: Path, start: float, end: float, signal_start: float, signal_end: float, turns: list[dict]) -> dict:
    if end <= start:
        raise ValueError("Interval end must follow start")
    if (session_dir / "signal_meta.json").exists():
        meta, times, signals = window(session_dir, signal_start, signal_end)
        by_name = {name: signals[:, i] for i, name in enumerate(meta["channels"])} if len(times) else {}
        fs = float(meta.get("sample_rate_hz") or 0)
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0
    else:
        times, by_name, fs, duration = [], {}, 0, 0
    ecg = _heart_stats(by_name["ECG"], fs, "ecg", duration) if "ECG" in by_name else None
    ppg = _heart_stats(by_name["PPG"], fs, "ppg", duration) if "PPG" in by_name else None
    video = _video_stats(session_dir, start, end)
    ecg_bpm = ecg.get("heart_rate_bpm") if ecg else None
    ppg_bpm = ppg.get("heart_rate_bpm") if ppg else None
    remote_bpm = video.get("rppg_heart_rate_bpm")
    return {
        "video_interval": [start, end],
        "signal_interval": [signal_start, signal_end],
        "sample_count": len(times),
        "contact": {
            "ecg": ecg,
            "ppg": ppg,
            "heart_rate_monitor_bpm": _mean(by_name.get("HeartRate", [])),
            "spo2_percent": _mean(by_name.get("SpO2", [])),
            "respiration_bpm": _respiration(by_name["RIP"], fs) if "RIP" in by_name else None,
            "gsr_mean": _mean(by_name.get("GSR", [])),
        },
        "video": video,
        "comparison": {
            "ecg_ppg_delta_bpm": round(abs(ecg_bpm - ppg_bpm), 1) if ecg_bpm is not None and ppg_bpm is not None else None,
            "ecg_rppg_delta_bpm": round(abs(ecg_bpm - remote_bpm), 1) if ecg_bpm is not None and remote_bpm is not None else None,
        },
        "speech": _speech_stats(turns, start, end),
        "lie_probability": None,
        "notes": ["ВСР рассчитывается только на интервале от 120 секунд."] if by_name and duration < 120 else [],
    }


@lru_cache(maxsize=2048)
def _hrv_at(directory: str, end_time: int, channel: str):
    session_dir = Path(directory)
    meta, times, signals = window(session_dir, end_time - 120, end_time)
    if len(times) < 120 * float(meta.get("sample_rate_hz") or 0) * 0.95:
        return None
    if channel not in meta["channels"]:
        return None
    values = signals[:, meta["channels"].index(channel)]
    return _heart_stats(values, float(meta["sample_rate_hz"]), "ecg" if channel == "ECG" else "ppg", float(times[-1] - times[0]))["lf_hf"]


def sliding_hrv(session_dir: Path, start: float, end: float) -> dict:
    """120 second rolling LF/HF, evaluated every 5 seconds."""
    meta = json.loads((session_dir / "signal_meta.json").read_text(encoding="utf-8"))
    first = max(int(np.ceil(start / 5) * 5), int(np.ceil((meta["start"] + 120) / 5) * 5))
    last = min(int(np.floor(end / 5) * 5), int(meta["end"]))
    result = {"time": [], "ecg_lf_hf": [], "ppg_lf_hf": []}
    for point in range(first, last + 1, 5):
        result["time"].append(point)
        result["ecg_lf_hf"].append(_hrv_at(str(session_dir), point, "ECG") if "ECG" in meta["channels"] else None)
        result["ppg_lf_hf"].append(_hrv_at(str(session_dir), point, "PPG") if "PPG" in meta["channels"] else None)
    return result


def clear_hrv_cache() -> None:
    _hrv_at.cache_clear()
