"""Speech summaries with dialogue context and explicit boundary limitations."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


def _tokens(text):
    return re.findall(r"[\wа-яё]+", text or "", re.I)


def _union_length(spans):
    merged = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def _answers(turns):
    """Infer exchanges from consecutive speaker runs, not ASR segment boundaries."""
    blocks = []
    for turn in sorted(turns, key=lambda t: t["start"]):
        if blocks and blocks[-1]["speaker"] == turn.get("speaker"):
            blocks[-1]["turns"].append(turn)
            blocks[-1]["end"] = max(blocks[-1]["end"], turn["end"])
        else:
            blocks.append(dict(speaker=turn.get("speaker"), start=turn["start"], end=turn["end"], turns=[turn]))
    for i, block in enumerate(blocks):
        if block["speaker"] == "respondent":
            block["latency"] = block["start"] - blocks[i - 1]["end"] if i and blocks[i - 1]["speaker"] == "interviewer" else None
            yield block


@lru_cache(maxsize=4)
def load_audio_frames(path, modified_ns):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def speech_stats(turns, start, end, audio_frames=None):
    selected = [t for t in turns if t.get("end", 0) > start and t.get("start", 0) < end]
    respondent = [t for t in selected if t.get("speaker") == "respondent"]
    answers = [a for a in _answers(turns) if any(t in respondent for t in a["turns"])]
    response_seconds = _union_length([(max(start, a["start"]), min(end, a["end"])) for a in answers])
    segment_seconds = _union_length([(max(start, t["start"]), min(end, t["end"])) for t in respondent])
    text_parts, word_method, count_available = [], "whole_turns", bool(respondent)
    for turn in respondent:
        if turn["start"] >= start - 1e-6 and turn["end"] <= end + 1e-6:
            text_parts.append(turn.get("text", ""))
        elif turn.get("words"):
            word_method = "word_timestamps"
            text_parts.extend(w.get("word", "") for w in turn["words"] if w["start"] >= start and w["end"] <= end)
        else:
            count_available = False
    words = sum(len(_tokens(text)) for text in text_parts) if count_available else None
    syllables = sum(len(re.findall(r"[аеёиоуыэюяaeiouy]", text, re.I)) for text in text_parts) if count_available else None
    latencies = [a["latency"] for a in answers if a["latency"] is not None]
    pauses = []
    pauses_complete = bool(respondent)
    for turn in respondent:
        if turn.get("words"):
            pairs = zip(turn["words"][:-1], turn["words"][1:])
            pauses.extend(max(0, min(end, right["start"]) - max(start, left["end"])) for left, right in pairs if right["start"] - left["end"] >= 0.25)
        elif turn["start"] >= start and turn["end"] <= end and turn.get("pause_seconds") is not None:
            pauses.append(float(turn["pause_seconds"]))
        else:
            pauses_complete = False
    for answer in answers:
        for left, right in zip(answer["turns"][:-1], answer["turns"][1:]):
            gap = max(0, min(end, right["start"]) - max(start, left["end"]))
            if gap >= 0.25:
                pauses.append(gap)
    result = dict(respondent_words=words,
                  words_per_minute=words * 60 / response_seconds if words is not None and response_seconds else None,
                  syllables_per_second=syllables / response_seconds if syllables is not None and response_seconds else None,
                  word_count_method=word_method if count_available else ("unavailable_partial_turn" if respondent else "no_respondent_speech"),
                  response_latency_seconds=float(np.mean(latencies)) if latencies else None,
                  latency_count=len(latencies), latency_method="consecutive_speaker_runs",
                  turn_count=len(selected), answer_count=len(answers),
                  response_duration_seconds=float(np.mean([a["end"] - a["start"] for a in answers])) if answers else None,
                  response_duration_scope="whole_answer",
                  internal_pause_seconds=sum(pauses) if pauses_complete else None,
                  pause_ratio=sum(pauses) / response_seconds if pauses_complete and response_seconds else None,
                  voice_rms=None, pitch_hz=None, pitch_iqr_hz=None, spectral_centroid_hz=None,
                  acoustic_method="unavailable", acoustic_coverage=0.0)
    if audio_frames is not None and respondent:
        mask = np.zeros(len(audio_frames["start"]), dtype=bool)
        for turn in respondent:
            mask |= (audio_frames["start"] >= max(start, turn["start"]) - 1e-6) & (audio_frames["end"] <= min(end, turn["end"]) + 1e-6)
        if mask.any():
            centers = (audio_frames["start"][mask] + audio_frames["end"][mask]) / 2
            _, unique = np.unique(centers, return_index=True)
            rms = audio_frames["rms"][mask][unique]
            pitch = audio_frames["pitch_hz"][mask][unique]
            pitch = pitch[np.isfinite(pitch)]
            result.update(voice_rms=float(np.sqrt(np.mean(rms ** 2))),
                          pitch_hz=float(np.median(pitch)) if len(pitch) else None,
                          pitch_iqr_hz=float(np.percentile(pitch, 75) - np.percentile(pitch, 25)) if len(pitch) else None,
                          spectral_centroid_hz=float(np.mean(audio_frames["spectral_centroid_hz"][mask][unique])),
                          acoustic_method="frames_v2", acoustic_coverage=min(1.0, _union_length(list(zip(audio_frames["start"][mask], audio_frames["end"][mask]))) / segment_seconds) if segment_seconds else 0.0)
    elif respondent and all(t["start"] >= start and t["end"] <= end and t.get("acoustic", {}).get("version") == 2 for t in respondent):
        # Duration weighting of full-turn RMS is exact for nonoverlapping turns.
        weighted = [(t["end"] - t["start"], t.get("acoustic", {}).get("rms")) for t in respondent]
        weighted = [(weight, value) for weight, value in weighted if value is not None]
        if weighted:
            result["voice_rms"] = float(np.sqrt(sum(w * v * v for w, v in weighted) / sum(w for w, _ in weighted)))
            result["acoustic_method"] = "full_turn_rms_v2"
    return result
