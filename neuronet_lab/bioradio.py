"""Streaming import for BioRadio CSV exports, including decimal-comma exports."""

from __future__ import annotations

import csv
import json
import re
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
IMPORT_VERSION = 2
DECIMAL_FIELDS = {"ECG", "RIP", "GSR", "Flow Rate temp", "PPG pulse"}


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
    """Recover unquoted decimal commas, including integer analog values.

    This export is intrinsically ambiguous in some rows. Enumerate legal
    layouts and reject ambiguity rather than silently shifting channels.
    Device HR/SpO2/Event are integer anchors in this particular format.
    Reconstruction accepts analog magnitudes up to 2**31; this explicit
    format bound excludes decimal tails accidentally read as whole values.
    Ordinary and correctly quoted CSVs do not use these restrictions.
    """
    headers, fields = list(headers), list(fields)
    while headers and not headers[-1]:
        headers.pop()
        if fields and not fields[-1].strip():
            fields.pop()
    names, tokens = headers[1:], [s.strip() for s in fields[1:]]
    solutions = []

    def visit(index, cursor, values):
        if len(solutions) > 1:
            return
        if index == len(names):
            if cursor == len(tokens):
                solutions.append(values)
            return
        remaining = len(names) - index
        if not remaining <= len(tokens) - cursor <= remaining + sum(n in DECIMAL_FIELDS for n in names[index:]):
            return
        name = names[index]
        raw = tokens[cursor]
        choices = [(raw, 1)]
        if name in DECIMAL_FIELDS and cursor + 1 < len(tokens):
            fraction = tokens[cursor + 1]
            if re.fullmatch(r"[+-]?\d+", raw) and re.fullmatch(r"\d+(?:[eE][+-]?\d+)?", fraction):
                choices.append((raw + "." + fraction, 2))
        for raw_value, width in choices:
            value = _number(raw_value)
            if raw_value and not np.isfinite(value):
                continue
            if name in DECIMAL_FIELDS and np.isfinite(value) and abs(value) > 2 ** 31:
                continue
            if name not in DECIMAL_FIELDS and name in CHANNELS and np.isfinite(value):
                maximum = {"Heart Rate pulse": 300, "SpO2 pulse": 100, "BioRadio Event": 65535}.get(name)
                if maximum is not None and (value != int(value) or not 0 <= value <= maximum):
                    continue
            visit(index + 1, cursor + width, values + [value])

    visit(0, 0, [])
    if len(solutions) != 1:
        raise ValueError("Ambiguous or malformed unquoted decimal-comma row")
    return {CHANNELS[name]: value for name, value in zip(names, solutions[0]) if name in CHANNELS}


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
        columns = {name: array("d") for name in names}
        times = array("d")
        malformed = 0
        recovered = 0
        mode = None
        for fields in reader:
            if not fields or not fields[0].strip():
                continue
            try:
                t = parse_clock(fields[0])
                if not np.isfinite(t) or t < 0:
                    raise ValueError("Invalid sample time")
                split_decimals = delimiter == "," and len(fields) > len(headers)
                if split_decimals:
                    mode = "decimal_comma"
                    try:
                        values = _decimal_comma_row(fields, headers)
                        recovered += 1
                    except ValueError:
                        # Keep the timestamp and explicit gaps, never compress time.
                        values = {}
                        malformed += 1
                elif len(fields) in {len(headers), len(headers) - (not headers[-1])}:
                    mode = mode or "regular"
                    values = {
                        CHANNELS[name]: _number(fields[index])
                        for index, name in enumerate(headers)
                        if name in CHANNELS and index < len(fields)
                    }
                else:
                    values = {}
                    malformed += 1
                if times and t <= times[-1]:
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
    signal_array = np.column_stack([np.frombuffer(columns[name], dtype=np.float64) for name in names])
    for name, values in (("time.npy", time_array), ("signals.npy", signal_array)):
        temporary = output_dir / (name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values)
        temporary.replace(output_dir / name)
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
        "import_version": IMPORT_VERSION,
        "recovered_decimal_rows": recovered,
        "nonfinite_samples": {name: int(np.count_nonzero(~np.isfinite(signal_array[:, i]))) for i, name in enumerate(names)},
    }
    temporary = output_dir / "signal_meta.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_dir / "signal_meta.json")
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
