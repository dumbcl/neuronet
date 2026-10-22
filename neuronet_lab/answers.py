"""Build full response records from transcript roles and current expert intervals.

No text/voice classification: labels must cover the response in the supplied
annotations. Existing response groupings and verified pauses are reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid

from .episodes import EPISODE_VERSION, answer_episode


BUILD_VERSION = 1
LABELS = {"truth", "lie", "base", "unlabeled"}
EPS = .001


def _merge(spans):
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _length(spans):
    return sum(b-a for a, b in _merge(spans))


def _overlap(left, right):
    return max(0.0, min(left[1], right[1])-max(left[0], right[0]))


def _subtract(spans, cuts):
    result = []
    for start, end in _merge(spans):
        for a, b in _merge(cuts):
            if b <= start or a >= end:
                continue
            if a > start:
                result.append((start, a))
            start = max(start, b)
        if end > start:
            result.append((start, end))
    return result


def _new_id(prefix, kind):
    return f"{prefix}-{kind}-{uuid.uuid4().hex[:12]}"


def _validate_intervals(items, kind):
    seen = set()
    for item in items:
        start, end = item["start"], item["end"]
        if not (isinstance(start, (int, float)) and isinstance(end, (int, float)) and
                math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"{kind}: конец должен быть позже начала, время — в секундах от начала видео.")
        if item.get("id"):
            if item["id"] in seen:
                raise ValueError(f"{kind}: повторяется идентификатор {item['id']}.")
            seen.add(item["id"])


def _annotation_sources(meta, annotations, prefix):
    sources = copy.deepcopy(meta.get("annotation_sources") or [])
    known = {s["id"]: s for s in sources if s.get("id")}
    for item in annotations:
        core = {k: item.get(k) for k in ("start", "end", "label", "name")}
        source_id = item.get("annotation_source_id")
        source = known.get(source_id)
        if source is None or any(source.get(k) != v for k, v in core.items()):
            digest = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
            source_id = f"{prefix}-source-{digest}"
            if source_id not in known:
                source = dict(core, id=source_id, origin="current_annotations")
                sources.append(source)
                known[source_id] = source
        item["annotation_source_id"] = source_id
    return sources


def _groups(meta, turns):
    """Retain valid existing semantic groups, then infer unassigned speaker runs."""
    by_id = {t["id"]: t for t in turns}
    groups, retained_events, used = [], [], set()
    for old in meta.get("answers") or []:
        ids = old.get("turn_ids", [])
        if not ids:
            # An explicitly retained nonverbal/unverified event must not disappear.
            if old.get("id") and old.get("dataset_eligible") is False:
                event = copy.deepcopy(old)
                event.update(review_required=True, dataset_eligible=False)
                event.setdefault("review_reason", "Событие без надёжной речевой реплики; сохранено для проверки.")
                retained_events.append(event)
            continue
        if len(ids) != len(set(ids)) or any(k in used or k not in by_id or by_id[k]["speaker"] != "respondent" for k in ids):
            continue
        parts = sorted((by_id[k] for k in ids), key=lambda t: t["start"])
        start, end = parts[0]["start"], max(t["end"] for t in parts)
        # New respondent material inside the old group means its grouping is stale.
        if any(t["speaker"] == "respondent" and t["id"] not in ids and _overlap((start,end), (t["start"],t["end"])) > EPS for t in turns):
            continue
        groups.append((parts, old))
        used.update(ids)
    pending = []
    for t in turns:
        if t["speaker"] == "respondent" and t["id"] not in used:
            pending.append(t)
        elif pending:
            groups.append((pending, None))
            pending = []
    if pending:
        groups.append((pending, None))
    groups.sort(key=lambda g: g[0][0]["start"])
    return groups, retained_events


def _reviewed_pauses(old, start, end):
    old = old or {}
    source = old.get("reviewed_internal_pauses")
    if source is None:
        source = [] if old.get("builder_version") else old.get("internal_pauses", [])
    return [copy.deepcopy(p) for p in source if start <= p["start"] < p["end"] <= end]


def _pauses(parts, old, annotations):
    start, end = parts[0]["start"], max(t["end"] for t in parts)
    pauses = [(a["end"],b["start"]) for a,b in zip(parts[:-1],parts[1:]) if b["start"]-a["end"] >= .25]
    for t in parts:
        words = sorted(t.get("words") or [], key=lambda w: w["start"])
        pauses.extend((a["end"],b["start"]) for a,b in zip(words[:-1],words[1:])
                      if t["start"] <= a["end"] < b["start"] <= t["end"] and b["start"]-a["end"] >= .25)
    pauses.extend((p["start"],p["end"]) for p in _reviewed_pauses(old,start,end))
    binary = [a for a in annotations if a["label"] in {"truth", "lie"}]
    for a in annotations:
        if a["label"] != "base" or a.get("kind") == "rest" or not start < a["start"] < a["end"] < end:
            continue
        left = {b["label"] for b in binary if b["start"] < a["start"] and b["end"] > start}
        right = {b["label"] for b in binary if b["end"] > a["end"] and b["start"] < end}
        # An interior expert base between the same response class denotes a
        # respondent pause. Edge listening/rest is never relabelled as response.
        if a.get("kind") == "internal_pause" or len(left) == len(right) == 1 and left == right:
            pauses.append((a["start"],a["end"]))
    return [dict(start=a,end=b) for a,b in _merge(pauses)]


def build_full_answers(meta):
    """Return session changes plus a report; do not mutate the caller or files."""
    turns = copy.deepcopy(meta.get("turns") or [])
    annotations = copy.deepcopy(meta.get("annotations") or [])
    if not turns or not any(t.get("speaker") == "respondent" for t in turns):
        raise ValueError("Сначала добавьте стенограмму и укажите реплики респондента.")
    if not annotations:
        raise ValueError("Сначала сохраните интервалы с экспертными метками во вкладке «Разметка».")
    _validate_intervals(turns, "Стенограмма")
    _validate_intervals(annotations, "Разметка")
    if any(t.get("speaker") not in {"interviewer", "respondent"} for t in turns):
        raise ValueError("В стенограмме допустимы только Интервьюер и Респондент.")
    if any(a.get("label") not in LABELS for a in annotations):
        raise ValueError("Неизвестная экспертная метка интервала.")
    prefix = meta["id"][:8]
    for items, kind in ((turns,"turn"), (annotations,"interval")):
        for item in items:
            if not item.get("id"):
                item["id"] = _new_id(prefix, kind)
    turns.sort(key=lambda t: t["start"])
    annotations.sort(key=lambda a: a["start"])
    sources = _annotation_sources(meta, annotations, prefix)
    groups, events = _groups(meta, turns)
    answers, issues = [], []
    positions = {t["id"]: i for i,t in enumerate(turns)}
    for parts, old in groups:
        start, end = parts[0]["start"], max(t["end"] for t in parts)
        pauses = _pauses(parts, old, annotations)
        support = _subtract([(t["start"],t["end"]) for t in parts], [(p["start"],p["end"]) for p in pauses])
        hits = [a for a in annotations if any(_overlap((a["start"],a["end"]),s) > EPS for s in support)]
        binary = [a for a in hits if a["label"] in {"truth", "lie"}]
        labels = {a["label"] for a in binary}
        covered = _length([(max(a["start"],s[0]),min(a["end"],s[1])) for a in binary for s in support])
        missing = max(0.0, _length(support)-covered)
        reasons = []
        label = next(iter(labels)) if len(labels) == 1 else "unlabeled"
        if len(labels) > 1:
            reasons.append("В одном полном ответе пересекаются метки «правда» и «ложь».")
        elif not labels:
            base_covered = _length([(max(a["start"],s[0]),min(a["end"],s[1])) for a in hits if a["label"] == "base" for s in support])
            if support and base_covered >= _length(support)-EPS:
                label = "base"
            else:
                reasons.append("Нет полной экспертной метки «правда/ложь» для речи ответа.")
        elif missing > EPS:
            reasons.append(f"Часть речи не покрыта меткой «правда/ложь»: {missing:.3f} с.")
        if any(a["label"] == "unlabeled" for a in hits):
            reasons.append("В речи ответа есть явно неразмеченный интервал.")
        if any(t.get("review_kind") in {"clarification", "technical", "uncertain_speaker"} for t in parts):
            reasons.append("Уточняющая/техническая реплика или непроверенная роль говорящего.")
        if any(a.get("review_required") and a.get("label") in {"truth", "lie"} for a in hits):
            reasons.append("Экспертный интервал помечен как требующий проверки.")
        if old and not old.get("builder_version") and old.get("review_reason") and (old.get("review_required") or old.get("dataset_eligible") is False):
            reasons.append(old["review_reason"])
        if reasons:
            label = "unlabeled"
        question = []
        i = positions[parts[0]["id"]]-1
        while i >= 0 and turns[i]["speaker"] == "interviewer":
            question.append(turns[i]); i -= 1
        question.reverse()
        source_ids = [a["annotation_source_id"] for a in binary if a["label"] == label]
        if not source_ids:
            source_ids = [a["annotation_source_id"] for a in hits]
        answer = dict(old or {}, id=(old or {}).get("id") or _new_id(prefix,"answer"),
                      participant_id=(old or {}).get("participant_id") or meta.get("participant_id") or meta["id"],
                      start=start,end=end,label=label,turn_ids=[t["id"] for t in parts],
                      question_turn_ids=[t["id"] for t in question],
                      question_start=min(t["start"] for t in question) if question else None,
                      question_end=max(t["end"] for t in question) if question else None,
                      source_annotation_ids=list(dict.fromkeys(source_ids)),internal_pauses=pauses,
                      reviewed_internal_pauses=_reviewed_pauses(old,start,end),
                      dataset_eligible=label in {"truth","lie"} and not reasons,
                      review_required=bool(reasons),builder_version=BUILD_VERSION)
        answer.pop("review_reason", None)
        if reasons:
            answer["review_reason"] = " ".join(dict.fromkeys(reasons))
        answers.append(answer)
    answers.extend(events)
    answers.sort(key=lambda a: a["start"])
    answer_ids = [a["id"] for a in answers]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Повторяются идентификаторы полных ответов; проверьте существующие связи.")
    for i, answer in enumerate(answers):
        overlaps = [a for a in answers[:i] if a.get("turn_ids") and answer.get("turn_ids") and a["end"] > answer["start"]+EPS]
        if overlaps:
            for a in [answer,*overlaps]:
                a.update(dataset_eligible=False,review_required=True,review_reason="Полные ответы пересекаются по времени; проверьте границы реплик.")
    episode_meta = dict(meta, turns=turns, annotations=annotations, answers=answers)
    for answer in answers:
        answer["episode"] = answer_episode(episode_meta,answer)
        if answer.get("review_required"):
            issues.append(dict(answer_id=answer["id"],start=answer["start"],end=answer["end"],reason=answer.get("review_reason","Требуется проверка.")))
    for a in annotations:
        containing = [x for x in answers if x["start"]-EPS <= a["start"] and a["end"] <= x["end"]+EPS]
        a.pop("answer_id",None); a.pop("answer_label",None)
        if len(containing) == 1:
            parent = containing[0]
            a["answer_id"] = parent["id"]
            if a["label"] == "base":
                a["answer_label"] = parent["label"]
    unlinked = [a for a in annotations if a["label"] in {"truth","lie"} and not any(
        t["speaker"] == "respondent" and _overlap((a["start"],a["end"]),(t["start"],t["end"])) > EPS for t in turns)]
    report = dict(version=BUILD_VERSION,answer_count=len(answers),eligible_count=sum(a["dataset_eligible"] for a in answers),
                  review_count=sum(bool(a.get("review_required")) for a in answers),
                  base_count=sum(a["label"] == "base" for a in answers),issues=issues,
                  unlinked_intervals=[dict(id=a["id"],start=a["start"],end=a["end"],reason="Экспертный интервал не пересекает реплики респондента.") for a in unlinked],
                  grouping="existing_groups_then_consecutive_respondent_turns")
    changes = dict(turns=turns,annotations=annotations,annotation_sources=sources,answers=answers,
                   answer_review_required=False,episode_schema_version=EPISODE_VERSION)
    return changes, report
