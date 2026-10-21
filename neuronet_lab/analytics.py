"""Interval metrics. Values without enough signal are explicitly unavailable."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .bioradio import window
from .signal_processing import heart_stats as _heart_stats, pulse_peaks as _peaks, respiration as _respiration
from .speech import speech_stats as _speech_stats, load_audio_frames
from .video_features import video_stats as _video_stats


def _finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(values, digits=None):
    values = _finite(values)
    if not len(values):
        return None
    result = float(np.mean(values))
    return round(result, digits) if digits is not None else result


def _device_mean(values, maximum):
    values = _finite(values)
    return _mean(values[(values > 0) & (values <= maximum)])


def interval_stats(session_dir: Path, start: float, end: float, signal_start: float, signal_end: float, turns: list[dict], answer_spans=None, *, include_contact=True) -> dict:
    if end <= start:
        raise ValueError("Interval end must follow start")
    if include_contact and (session_dir / "signal_meta.json").exists():
        meta, times, signals = window(session_dir, signal_start, signal_end)
        by_name = {name: signals[:, i] for i, name in enumerate(meta["channels"])} if len(times) else {}
        fs = float(meta.get("sample_rate_hz") or 0)
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0
    else:
        times, by_name, fs, duration = [], {}, 0, 0
    ecg = _heart_stats(by_name["ECG"], fs, "ecg", duration, times) if "ECG" in by_name else None
    ppg = _heart_stats(by_name["PPG"], fs, "ppg", duration, times) if "PPG" in by_name else None
    video = _video_stats(session_dir, start, end)
    ecg_bpm = ecg.get("heart_rate_bpm") if ecg else None
    ppg_bpm = ppg.get("heart_rate_bpm") if ppg else None
    remote_bpm = video.get("rppg_heart_rate_bpm")
    audio_path = session_dir / "audio_features.npz"
    frames = load_audio_frames(audio_path, audio_path.stat().st_mtime_ns) if audio_path.exists() else None
    speech = _speech_stats(turns, start, end, frames, answer_spans)
    notes = []
    if by_name and duration + (1 / fs if fs else 0) < 120:
        notes.append("Для ВСР нужно не менее 120 секунд; ULF/VLF на таком окне недоступны.")
    if any(h and h["quality"] not in {"ok", "insufficient_duration"} for h in (ecg, ppg)):
        notes.append("Недостаточно качественных межударных интервалов: часть показателей ВСР недоступна.")
    if by_name and meta.get("import_version", 1) < 2:
        notes.append("Старый импорт BioRadio: повторно импортируйте исходный CSV.")
    if video["rppg_heart_rate_bpm"] is None and video.get("rppg_context_heart_rate_bpm") is not None:
        notes.append("Окно rPPG выходит за границы интервала; доступна только контекстная ЧСС.")
    if video.get("legacy_pose"):
        notes.append("Для обновления масштаба движений и наклона корпуса повторите анализ видео.")
    if speech["word_count_method"] == "unavailable_partial_turn":
        notes.append("Границы пересекают реплику без времён слов: темп речи не вычислен.")
    if speech["answer_count"]:
        notes.append("Длительность полного ответа включает внутренние паузы." if answer_spans is not None else "Задержка и полная длительность ответа определены по последовательным репликам говорящих.")
    if speech["acoustic_method"] == "unavailable" and speech["answer_count"]:
        notes.append("Для акустики пересчитайте её по видео во вкладке стенограммы.")
    gsr = _finite(by_name.get("GSR", []))
    gsr_unavailable = bool(len(gsr) and np.all(gsr == 100000))
    if gsr_unavailable:
        notes.append("GSR постоянен (100000): канал недоступен для анализа изменений.")
    return {
        "calculation_version": 2,
        "video_interval": [start, end],
        "signal_interval": [signal_start, signal_end],
        "sample_count": len(times),
        "contact": {
            "ecg": ecg,
            "ppg": ppg,
            "heart_rate_monitor_bpm": _device_mean(by_name.get("HeartRate", []), 300),
            "spo2_percent": _device_mean(by_name.get("SpO2", []), 100),
            "respiration_bpm": _respiration(by_name["RIP"], fs, times) if "RIP" in by_name else None,
            "gsr_mean": None if gsr_unavailable else _mean(gsr),
        },
        "video": video,
        "comparison": {
            "ecg_ppg_delta_bpm": abs(ecg_bpm - ppg_bpm) if ecg_bpm is not None and ppg_bpm is not None else None,
            "ecg_rppg_delta_bpm": abs(ecg_bpm - remote_bpm) if ecg_bpm is not None and remote_bpm is not None else None,
        },
        "speech": speech,
        "lie_probability": None,
        "notes": notes,
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
    return _heart_stats(values, float(meta["sample_rate_hz"]), "ecg" if channel == "ECG" else "ppg", float(times[-1] - times[0]), times)["lf_hf"]


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
