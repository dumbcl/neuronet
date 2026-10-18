"""Streaming import for BioRadio CSV exports, including decimal-comma exports."""

from __future__ import annotations

import csv
import json
from array import array
from pathlib import Path

import numpy as np


CHANNELS = {
    "ECG": "ECG",
    "RIP": "RIP",
    "GSR": "GSR",
    "Flow Rate temp": "Flow",
    "Heart Rate pulse": "HeartRate",
    "PPG pulse": "PPG",
    "SpO2 pulse": "SpO2",
    "BioRadio Event": "Event",
}


def parse_clock(value: str) -> float:
    value = value.strip()
    if ":" not in value:
        return float(value.replace(",", "."))
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time value: {value}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2].replace(",", "."))


def _number(value: str) -> float:
    try:
        return float(value.strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _decimal_comma_row(fields: list[str], headers: list[str]) -> dict[str, float]:
    """Rejoin split decimal values in BioRadio's comma-delimited EU export.

    In the supplied export, ECG/RIP/GSR/Flow/PPG each occupy two CSV tokens,
    while Heart Rate, SpO2 and Event occupy one. This is detected by row width.
    """
    decimal_fields = {"ECG", "RIP", "GSR", "Flow Rate temp", "PPG pulse"}
    cursor = 1
    values: dict[str, float] = {}
    for header in headers[1:]:
        if not header:
            continue
        if header in decimal_fields:
            raw = ",".join(fields[cursor : cursor + 2])
            cursor += 2
        else:
            raw = fields[cursor] if cursor < len(fields) else ""
            cursor += 1
        if header in CHANNELS:
            values[CHANNELS[header]] = _number(raw)
    return values


def import_csv(source: Path, output_dir: Path) -> dict:
    """Parse a large export once and save memory-mappable NumPy arrays.

    No DataFrame of individual samples is built, so a 25 minute / 1 kHz
    recording stays within a reasonable memory footprint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first_line = handle.readline()
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        handle.seek(0)
        reader = csv.reader(handle, delimiter=delimiter)
        headers = [name.strip() for name in next(reader)]
        if not headers or headers[0].lower() not in {"elapsed time", "time", "timestamp"}:
            raise ValueError("CSV needs an 'Elapsed Time' column")
        names = [CHANNELS[name] for name in headers if name in CHANNELS]
        if not names:
            raise ValueError("No recognized BioRadio channels in CSV")
        columns = {name: array("f") for name in names}
        times = array("d")
        malformed = 0
        mode = None
        for fields in reader:
            if not fields or not fields[0].strip():
                continue
            try:
                t = parse_clock(fields[0])
                if mode is None:
                    # Empty trailing header and data token are both retained.
                    mode = "decimal_comma" if delimiter == "," and len(fields) > len(headers) else "regular"
                if mode == "decimal_comma":
                    values = _decimal_comma_row(fields, headers)
                else:
                    values = {
                        CHANNELS[name]: _number(fields[index])
                        for index, name in enumerate(headers)
                        if name in CHANNELS and index < len(fields)
                    }
                if times and t < times[-1]:
                    malformed += 1
                    continue
                times.append(t)
                for name in names:
                    columns[name].append(values.get(name, float("nan")))
            except (ValueError, IndexError):
                malformed += 1
        if not times:
            raise ValueError("CSV contains no valid samples")

    time_array = np.frombuffer(times, dtype=np.float64).copy()
    signal_array = np.column_stack([np.frombuffer(columns[name], dtype=np.float32) for name in names])
    np.save(output_dir / "time.npy", time_array)
    np.save(output_dir / "signals.npy", signal_array)
    dt = np.diff(time_array[: min(len(time_array), 10001)].astype(np.float64))
    positive_dt = dt[dt > 0]
    sample_rate = round(1 / float(np.median(positive_dt)), 2) if len(positive_dt) else None
    summary = {
        "channels": names,
        "sample_count": len(time_array),
        "sample_rate_hz": sample_rate,
        "start": float(time_array[0]),
        "end": float(time_array[-1]),
        "duration": float(time_array[-1] - time_array[0]),
        "malformed_rows": malformed,
        "format": mode,
    }
    (output_dir / "signal_meta.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return summary


def load_arrays(session_dir: Path):
    meta = json.loads((session_dir / "signal_meta.json").read_text(encoding="utf-8"))
    times = np.load(session_dir / "time.npy", mmap_mode="r")
    signals = np.load(session_dir / "signals.npy", mmap_mode="r")
    return meta, times, signals


def window(session_dir: Path, start: float, end: float):
    meta, times, signals = load_arrays(session_dir)
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    return meta, np.asarray(times[left:right]), np.asarray(signals[left:right])


def plot_data(session_dir: Path, start: float, end: float, max_points: int = 1200) -> dict:
    meta, times, signals = window(session_dir, start, end)
    if not len(times):
        return {"time": [], "channels": {name: [] for name in meta["channels"]}}
    stride = max(1, int(np.ceil(len(times) / max_points)))
    times = times[::stride]
    signals = signals[::stride]
    return {
        "time": [round(float(x), 3) for x in times],
        "channels": {
            name: [round(float(v), 5) if np.isfinite(v) else None for v in signals[:, i]]
            for i, name in enumerate(meta["channels"])
        },
    }
