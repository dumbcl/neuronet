"""Export one question → waiting → full answer episode per reviewed target."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analytics import interval_stats
from .bioradio import window
from .episodes import EPISODE_VERSION, answer_episode
from .session import MARKER_NAMES, map_time


SCHEMA_VERSION = 2
CHANNELS = ("ECG", "PPG", "RIP", "GSR", "Flow", "HeartRate", "SpO2")
# Stable columns across sessions, including sessions with missing modalities.
FEATURES = {
    "contact.heart_rate_monitor_bpm": "bpm",
    "contact.spo2_percent": "%",
    "contact.respiration_bpm": "breaths/min",
    "contact.gsr_mean": "BioRadio source units",
    **{f"contact.{kind}.{key}": unit for kind in ("ecg", "ppg") for key, unit in (
        ("heart_rate_bpm", "bpm"), ("sdnn_ms", "ms"), ("rmssd_ms", "ms"),
        ("lf_power", "ms²"), ("hf_power", "ms²"), ("total_power", "ms², 0.04–0.4 Hz"), ("lf_hf", "ratio"))},
    **{f"speech.{key}": unit for key, unit in (
        ("respondent_words", "count"), ("words_per_minute", "words/min"),
        ("syllables_per_second", "syllables/s"), ("response_latency_seconds", "s"),
        ("response_duration_seconds", "s"), ("internal_pause_seconds", "s"), ("pause_ratio", "ratio"),
        ("voice_rms", "normalized audio amplitude"), ("pitch_hz", "Hz"),
        ("pitch_iqr_hz", "Hz"), ("spectral_centroid_hz", "Hz"))},
    **{f"video.{key}": unit for key, unit in (
        ("rppg_heart_rate_bpm", "bpm; complete window inside answer"),
        ("head_motion", "face widths/s"), ("hand_motion", "image heights/s; see motion_units"),
        ("leg_motion", "image heights/s; see motion_units"), ("trunk_tilt_degrees", "degrees"))},
    "comparison.ecg_ppg_delta_bpm": "bpm",
    "comparison.ecg_rppg_delta_bpm": "bpm",
}
QUALITY_PATHS = (
    "contact.ecg.quality", "contact.ecg.rr_count", "contact.ecg.rejected_rr_count",
    "contact.ppg.quality", "contact.ppg.rr_count", "contact.ppg.rejected_rr_count",
    "speech.word_count_method", "speech.latency_method", "speech.pause_measurement",
    "speech.acoustic_method", "speech.acoustic_coverage",
    "video.quality", "video.rppg_coverage", "video.rppg_window_count", "video.motion_units",
    "video.landmark_coverage.head_motion", "video.landmark_coverage.hand_motion",
    "video.landmark_coverage.leg_motion", "video.landmark_coverage.trunk_tilt",
)
IDENTIFIERS = (
    "dataset_schema_version", "calculation_version", "participant_id", "session_id", "session_name",
    "answer_id", "target", "label", "video_start_seconds", "video_end_seconds", "answer_duration_seconds",
    "signal_start_seconds", "signal_end_seconds", "question_start_seconds", "question_end_seconds",
    "respondent_text", "question_text", "turn_ids", "source_annotation_ids", "internal_pauses",
    "sync_status", "sync_uncertainty_seconds", "signal_sample_count", "signal_time_coverage",
)


def _column(path):
    return path.replace(".", "_")


PHASES = ("question", "waiting", "episode")
PHYSIOLOGY_FEATURES = {path: unit.replace("inside answer", "inside this phase") for path, unit in FEATURES.items() if not path.startswith("speech.")}
PHYSIOLOGY_QUALITY = tuple(path for path in QUALITY_PATHS if not path.startswith("speech."))
PHASE_COLUMNS = tuple(column for phase in PHASES for column in (
    f"{phase}_signal_start_seconds", f"{phase}_signal_end_seconds", f"{phase}_signal_sample_count", f"{phase}_signal_time_coverage",
    *(f"{phase}_{_column(path)}" for path in (*PHYSIOLOGY_FEATURES, *PHYSIOLOGY_QUALITY)),
    *(f"{phase}_signal_{c.lower()}_valid_coverage" for c in CHANNELS)))
DELTA_FEATURES = {f"delta_answer_minus_question_{_column(path)}": unit for path, unit in PHYSIOLOGY_FEATURES.items()}
EPISODE_COLUMNS = (
    "episode_start_seconds", "episode_end_seconds", "episode_duration_seconds", "episode_status",
    "answer_start_seconds", "answer_end_seconds", "question_phase_start_seconds", "question_phase_end_seconds",
    "question_duration_seconds", "waiting_start_seconds", "waiting_end_seconds", "waiting_duration_seconds",
    "question_overlap_seconds", "episode_question_turn_ids",
)
COLUMNS = (*IDENTIFIERS[:8], *EPISODE_COLUMNS, *IDENTIFIERS[8:], *map(_column, FEATURES), *map(_column, QUALITY_PATHS),
           *(f"signal_{c.lower()}_valid_coverage" for c in CHANNELS), *PHASE_COLUMNS, *DELTA_FEATURES, "notes")


def _get(data, path):
    for key in path.split("."):
        data = data.get(key) if isinstance(data, dict) else None
    return data


def _clean(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def dataset_filename(name):
    name = name.strip()
    if name.lower().endswith(".csv"):
        name = name[:-4]
    if (not name or name.startswith(".") or name.endswith((" ", ".")) or len(name) > 120 or
            re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or
            name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Введите имя файла без пути и символов / \\ : * ? \" < > | (до 120 символов).")
    return f"{name}.csv"


def _validated_answers(meta):
    if meta.get("answer_review_required"):
        raise ValueError("После изменения разметки нужно проверить полные ответы: их границы или классы могли устареть. Выгрузка отменена.")
    if not meta.get("answers"):
        raise ValueError("В сессии нет размеченных полных ответов (answers). Для датасета нужно связать реплики и внутренние паузы в полные ответы с метками правда/ложь.")
    turns = meta.get("turns", [])
    ids = [t.get("id") for t in turns if t.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("Идентификаторы реплик повторяются: проверьте стенограмму.")
    by_id = {t["id"]: t for t in turns if t.get("id")}
    selected, excluded, seen = [], [], set()
    for answer in sorted(meta["answers"], key=lambda a: a["start"]):
        aid = answer.get("id")
        if not aid or aid in seen:
            raise ValueError("Идентификаторы полных ответов отсутствуют или повторяются.")
        seen.add(aid)
        reason = None
        if answer.get("label") not in {"truth", "lie"}:
            reason = "Нет бинарного класса правда/ложь"
        elif answer.get("review_required") or not answer.get("dataset_eligible", False):
            reason = answer.get("review_reason") or "Ответ требует проверки или исключён из датасета"
        if reason:
            excluded.append(dict(answer_id=aid, reason=reason))
            continue
        parts = [by_id.get(tid) for tid in answer.get("turn_ids", [])]
        if not parts or any(t is None or t.get("speaker") != "respondent" for t in parts):
            raise ValueError(f"Ответ {aid}: повреждены связи с репликами респондента.")
        if len(answer["turn_ids"]) != len(set(answer["turn_ids"])):
            raise ValueError(f"Ответ {aid}: реплика указана несколько раз.")
        start, end = answer["start"], answer["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Ответ {aid}: некорректные границы.")
        if abs(start - min(t["start"] for t in parts)) > .001 or abs(end - max(t["end"] for t in parts)) > .001:
            raise ValueError(f"Ответ {aid}: границы не соответствуют репликам, проверьте полный ответ.")
        if any(q not in by_id or by_id[q]["speaker"] != "interviewer" for q in answer.get("question_turn_ids", [])):
            raise ValueError(f"Ответ {aid}: повреждены связи с вопросом интервьюера.")
        if any(not start <= p["start"] < p["end"] <= end for p in answer.get("internal_pauses", [])):
            raise ValueError(f"Ответ {aid}: пауза выходит за границы ответа.")
        if selected and start < selected[-1]["end"] - .001:
            raise ValueError("Полные ответы пересекаются: проверьте границы перед выгрузкой.")
        selected.append(answer)
    if not selected:
        raise ValueError("Нет проверенных полных ответов с метками правда/ложь для выгрузки.")
    return selected, excluded, by_id


def _sync(meta):
    markers = meta.get("markers", {})
    pairs = [(markers.get("video", {}).get(k), markers.get("signal", {}).get(k)) for k in MARKER_NAMES]
    count = sum(a is not None and b is not None for a, b in pairs)
    estimate = meta.get("sync_estimate") or {}
    if not count:
        return "missing", None
    if estimate.get("review_required"):
        return "estimated_needs_review", estimate.get("uncertainty_seconds")
    return ("single_anchor" if count == 1 else "markers_aligned"), estimate.get("uncertainty_seconds")


def _signal_coverage(directory, start, end, enabled):
    result = {f"signal_{c.lower()}_valid_coverage": None for c in CHANNELS}
    result.update(signal_sample_count=0, signal_time_coverage=0.0)
    if not enabled:
        return result
    meta, times, values = window(directory, start, end)
    expected = (end - start) * float(meta["sample_rate_hz"])
    result.update(signal_sample_count=len(times), signal_time_coverage=min(1.0, len(times) / expected) if expected else 0.0)
    for i, channel in enumerate(meta["channels"]):
        if channel in CHANNELS:
            valid = np.isfinite(values[:, i])
            if channel == "GSR" and valid.any() and np.all(values[valid, i] == 100000):
                valid[:] = False
            if channel in {"HeartRate", "SpO2"}:
                valid &= (values[:, i] > 0) & (values[:, i] <= (300 if channel == "HeartRate" else 100))
            result[f"signal_{channel.lower()}_valid_coverage"] = min(1.0, float(valid.sum()) / expected) if expected else 0.0
    return result


def _measure_context(directory, interval, markers, signal_enabled):
    if interval is None or interval["end"] <= interval["start"]:
        return {}, None, None, _signal_coverage(directory, None, None, False)
    start, end = interval["start"], interval["end"]
    left = map_time(start, markers["video"], markers["signal"]) if signal_enabled else None
    right = map_time(end, markers["video"], markers["signal"]) if signal_enabled else None
    # Context contributes physiology/video only, never speech or words from the interviewer.
    stats = interval_stats(directory, start, end, left, right, [], [], include_contact=signal_enabled)
    if not signal_enabled:
        stats["contact"], stats["comparison"] = {}, {}
    return stats, left, right, _signal_coverage(directory, left, right, signal_enabled)


def export_dataset(directory: Path, meta: dict, name: str, progress=lambda value: None):
    """Write a CSV and companion metadata from a fixed session snapshot.

    Missing sync suppresses contact features, but does not discard speech/video.
    Existing exports are retained using a numbered suffix.
    """
    filename = dataset_filename(name)
    answers, excluded, by_id = _validated_answers(meta)
    sync_status, uncertainty = _sync(meta)
    signal_present = all((directory / f).is_file() for f in ("signal_meta.json", "time.npy", "signals.npy"))
    signal_enabled = signal_present and sync_status != "missing"
    markers = meta["markers"]
    rows, episodes = [], []
    for index, answer in enumerate(answers):
        episode = answer_episode(meta, answer)
        episodes.append(dict(answer_id=answer["id"], **episode))
        start, end = answer["start"], answer["end"]
        left = map_time(start, markers["video"], markers["signal"]) if signal_enabled else None
        right = map_time(end, markers["video"], markers["signal"]) if signal_enabled else None
        # Only this response contributes speech features, even if ASR timings overlap a neighbour.
        context_ids = set(answer["turn_ids"] + answer.get("question_turn_ids", []))
        context = sorted((by_id[key] for key in context_ids), key=lambda t: t["start"])
        stats = interval_stats(directory, start, end, left, right, context, [answer], include_contact=signal_enabled)
        notes = list(stats["notes"]) + episode["notes"]
        if not signal_enabled:
            stats["contact"] = {}
            stats["comparison"] = {}
            notes.append("BioRadio не выгружен: " + ("нет синхронизации с видео." if sync_status == "missing" else "нет сохранённых сигналов."))
        elif sync_status == "estimated_needs_review":
            notes.append("Синхронизация BioRadio предварительная; требуется проверка.")
        speech = stats["speech"]
        if speech.get("response_latency_seconds") is not None and speech["response_latency_seconds"] < 0:
            speech["response_latency_seconds"] = None
            notes.append("Времена вопроса и ответа пересекаются; задержка ответа недоступна.")
        coverage = _signal_coverage(directory, left, right, signal_enabled)
        if signal_enabled and coverage["signal_time_coverage"] < .95:
            notes.append("BioRadio покрывает менее 95% длительности ответа.")
        row = dict(dataset_schema_version=SCHEMA_VERSION, calculation_version=stats["calculation_version"],
                   participant_id=answer.get("participant_id") or meta["id"], session_id=meta["id"], session_name=meta["name"],
                   answer_id=answer["id"], target=1 if answer["label"] == "lie" else 0, label=answer["label"],
                   video_start_seconds=start, video_end_seconds=end, answer_duration_seconds=end-start,
                   signal_start_seconds=left, signal_end_seconds=right,
                   question_start_seconds=answer.get("question_start"), question_end_seconds=answer.get("question_end"),
                   respondent_text=" ".join(by_id[key].get("text", "") for key in sorted(answer["turn_ids"], key=lambda k: by_id[k]["start"])),
                   question_text=" ".join(by_id[key].get("text", "") for key in episode["question_turn_ids"]),
                   turn_ids=json.dumps(answer["turn_ids"], ensure_ascii=False),
                   source_annotation_ids=json.dumps(answer.get("source_annotation_ids", []), ensure_ascii=False),
                   internal_pauses=json.dumps(answer.get("internal_pauses", []), ensure_ascii=False),
                   sync_status=sync_status, sync_uncertainty_seconds=uncertainty, **coverage)
        row.update({_column(path): _clean(_get(stats, path)) for path in (*FEATURES, *QUALITY_PATHS)})
        row.update(episode_start_seconds=episode["start"], episode_end_seconds=episode["end"],
                   episode_duration_seconds=episode["end"]-episode["start"], episode_status=episode["status"],
                   answer_start_seconds=start, answer_end_seconds=end,
                   question_overlap_seconds=episode["question_overlap_seconds"],
                   episode_question_turn_ids=json.dumps(episode["question_turn_ids"], ensure_ascii=False))
        phase_stats = {}
        for phase in PHASES:
            interval = episode if phase == "episode" else episode[phase]
            prefix = "question_phase" if phase == "question" else phase
            row[f"{prefix}_start_seconds"] = interval["start"] if interval else None
            row[f"{prefix}_end_seconds"] = interval["end"] if interval else None
            row[f"{phase}_duration_seconds"] = interval["end"]-interval["start"] if interval else None
            measured, phase_left, phase_right, phase_coverage = _measure_context(directory, interval, markers, signal_enabled)
            phase_stats[phase] = measured
            row.update({f"{phase}_{_column(path)}": _clean(_get(measured, path)) for path in (*PHYSIOLOGY_FEATURES, *PHYSIOLOGY_QUALITY)})
            row.update({f"{phase}_{key}": value for key, value in phase_coverage.items()})
            row[f"{phase}_signal_start_seconds"], row[f"{phase}_signal_end_seconds"] = phase_left, phase_right
            notes.extend(f"{phase}: {note}" for note in measured.get("notes", []))
            if signal_enabled and interval and interval["end"] > interval["start"] and phase_coverage["signal_time_coverage"] < .95:
                notes.append(f"{phase}: BioRadio покрывает менее 95% длительности интервала.")
        for path in PHYSIOLOGY_FEATURES:
            response, question = _clean(_get(stats, path)), _clean(_get(phase_stats["question"], path))
            row[f"delta_answer_minus_question_{_column(path)}"] = response-question if response is not None and question is not None else None
        row["notes"] = json.dumps(notes, ensure_ascii=False)
        rows.append(row)
        progress(int(95 * (index + 1) / len(answers)))

    target = directory / filename
    number = 2
    while target.exists() or target.with_suffix(".metadata.json").exists():
        target = directory / f"{Path(filename).stem} ({number}).csv"
        number += 1
    metadata_path = target.with_suffix(".metadata.json")
    manifest = {
        "dataset_schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_file": target.name, "encoding": "utf-8-sig", "delimiter": ",", "decimal_separator": ".",
        "missing_values": "empty CSV cells; never replaced with zero", "row_unit": "one question → waiting → full answer episode; target belongs to the answer",
        "target_mapping": {"truth": 0, "lie": 1}, "row_count": len(rows), "excluded_answers": excluded,
        "session_id": meta["id"], "session_name": meta["name"],
        "session_sha256": hashlib.sha256(json.dumps(meta, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "markers": markers, "sync_status": sync_status, "sync_estimate": meta.get("sync_estimate"),
        "signal_import": meta.get("signal"), "audio_analysis": meta.get("audio_analysis"), "video_analysis": meta.get("video_analysis"),
        "answer_snapshot": answers, "episode_version": EPISODE_VERSION, "episode_snapshot": episodes,
        "annotation_sources": meta.get("annotation_sources", []),
        "columns": list(COLUMNS), "numeric_feature_columns": {
            **{_column(path): unit for path, unit in FEATURES.items()},
            **{f"{phase}_{_column(path)}": unit for phase in PHASES for path, unit in PHYSIOLOGY_FEATURES.items()},
            **DELTA_FEATURES,
        },
        "window_scopes": {
            "unprefixed_features_and_video_start_end": "full answer only, retained from schema 1; answer_start/end are explicit aliases",
            "episode_": "question start through answer end, excluding earlier rest and earlier answers",
            "question_": "linked interviewer question, clipped before answer onset and after earlier rest/respondent speech",
            "waiting_": "question end to answer onset; absent when timings overlap, zero duration when immediately adjacent",
            "speech_": "respondent speech in the full answer only; internal pauses included in answer duration",
            "delta_answer_minus_question_": "answer feature minus question feature, only when both are available",
            "question_start_end_seconds": "original linked question bounds; question_phase_start/end are actual measured bounds",
        },
        "quality_columns": [*map(_column, QUALITY_PATHS), "sync_status", "sync_uncertainty_seconds", "signal_time_coverage",
                            *(f"signal_{c.lower()}_valid_coverage" for c in CHANNELS), "episode_status", "question_overlap_seconds",
                            *(c for c in PHASE_COLUMNS if c not in {f"{phase}_{_column(p)}" for phase in PHASES for p in PHYSIOLOGY_FEATURES}), "notes"],
        "research_notes": [
            "Split training and evaluation by participant_id. IDs, labels, source_annotation_ids and timing are not predictive features.",
            "Text and question content can disclose the interview protocol; assess text features separately to avoid target leakage.",
            "dataset_eligible means a reviewed target, not validated physiology. Filter features by sync_status and coverage.",
            "HRV requires at least 120 seconds and sufficient usable beats. Short answers retain missing HRV.",
            "Each rPPG value only uses windows fully inside its own phase; unrestricted context rPPG is excluded.",
            "GSR=100000 throughout the interval is unavailable. Channel coverage counts finite usable samples, not detector accuracy.",
            "Pauses may be incompletely annotated. Full answer duration always includes elapsed internal silence.",
            "Values are unrounded and not normalized to baseline. Baseline intervals and markers are in this metadata.",
            "Question/waiting physiology describes context associated with the answer; the interviewer and waiting are not independently labelled truth/lie.",
            "Epochs end at answer end. Post-answer silence and the next question are not added. Earlier rest is excluded.",
        ],
        "rest_intervals": meta.get("rest_intervals"),
        "baseline_annotations": [a for a in meta.get("annotations", []) if a.get("label") == "base"],
    }
    with tempfile.TemporaryDirectory(prefix=".dataset-", dir=directory) as temporary:
        staging = Path(temporary)
        with (staging / "data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (staging / "metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "metadata.json").replace(metadata_path)
        try:
            (staging / "data.csv").replace(target)
        except OSError:
            metadata_path.unlink(missing_ok=True)
            raise
    progress(100)
    return dict(csv_path=str(target), metadata_path=str(metadata_path), row_count=len(rows), excluded_count=len(excluded), sync_status=sync_status)
