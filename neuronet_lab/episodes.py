"""Derive question → waiting → answer windows from saved transcript links."""
from __future__ import annotations

import math


EPISODE_VERSION = 1


def _bounds(start, end):
    return isinstance(start, (int, float)) and isinstance(end, (int, float)) and math.isfinite(start) and math.isfinite(end) and 0 <= start < end


def answer_episode(meta, answer):
    """Keep the answer target and boundaries; add conservative temporal context.

    Rest and previous respondent speech form barriers. Overlapping ASR question
    timings are clipped for physiology; a negative latency is never made zero.
    """
    start, end = answer["start"], answer["end"]
    turns = {t["id"]: t for t in meta.get("turns", []) if t.get("id")}
    question = sorted((turns[k] for k in answer.get("question_turn_ids", [])
                       if k in turns and turns[k].get("speaker") == "interviewer"), key=lambda t: t["start"])
    raw_start = min(t["start"] for t in question) if question else answer.get("question_start")
    raw_end = max(t["end"] for t in question) if question else answer.get("question_end")
    episode = dict(version=EPISODE_VERSION, start=start, end=end,
                   answer=dict(start=start, end=end), question=None, waiting=None,
                   source_question_start=raw_start, source_question_end=raw_end,
                   question_turn_ids=[], question_overlap_seconds=0.0, status="answer_only", notes=[])
    if not _bounds(raw_start, raw_end) or raw_start >= start:
        episode["notes"].append("Нет надёжного интервала вопроса перед ответом; эпизод содержит только ответ.")
        return episode

    previous_end = max((t["end"] for t in meta.get("turns", []) if t.get("speaker") == "respondent"
                        and t.get("id") not in answer.get("turn_ids", []) and t["start"] < start), default=0)
    rests = list(meta.get("rest_intervals") or [])
    rests.extend(a for a in meta.get("annotations", []) if a.get("kind") == "rest" or
                 a.get("name") in {"Покой до беседы", "Покой после беседы"})
    rest_end = max((r["end"] for r in rests if r["start"] < start and r["end"] <= start), default=0)
    barrier = max(previous_end, rest_end)
    q_start, q_end = max(raw_start, barrier), min(raw_end, start)
    # A respondent interruption or a two-minute baseline can split an interviewer run.
    if question:
        question = [t for t in question if t["end"] > q_start and t["start"] < q_end]
        if question:
            q_start = max(q_start, min(t["start"] for t in question))
            q_end = min(q_end, max(t["end"] for t in question))
        else:
            q_end = q_start
    if not _bounds(q_start, q_end):
        episode["notes"].append("После исключения покоя и предыдущего ответа вопрос не выделяется; эпизод содержит только ответ.")
        return episode

    notes = episode["notes"]
    if q_start > raw_start + .001:
        notes.append("Начало вопроса ограничено границей покоя или предыдущего ответа.")
    overlap = max(0.0, raw_end - start)
    if overlap > 0:
        notes.append("Времена вопроса и ответа пересекаются; вопрос ограничен началом ответа, время ожидания недоступно.")
    episode.update(start=q_start, question=dict(start=q_start, end=q_end),
                   waiting=dict(start=q_end, end=start) if overlap == 0 else None,
                   question_turn_ids=[t["id"] for t in question], question_overlap_seconds=overlap,
                   status="question_overlap_clipped" if overlap else ("context_trimmed" if notes else "complete"))
    return episode
