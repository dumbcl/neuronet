"""On-disk sessions shared by the desktop UI and the analysis modules."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .analytics import clear_hrv_cache, interval_stats
from .bioradio import import_csv, plot_data
from .media import analyze_video, transcribe_and_diarize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sessions"
MARKER_NAMES = ("baseline_start", "dialog_start", "dialog_end", "post_end")
LOCK = threading.RLock()


def map_time(value: float, source: dict, target: dict) -> float:
    """Align video and BioRadio using corresponding stage boundaries."""
    pairs = sorted(
        (float(source[key]), float(target[key]))
        for key in MARKER_NAMES
        if source.get(key) is not None and target.get(key) is not None
    )
    if not pairs:
        return value
    if len(pairs) == 1:
        return value + pairs[0][1] - pairs[0][0]
    if value <= pairs[0][0]:
        left, right = pairs[:2]
    elif value >= pairs[-1][0]:
        left, right = pairs[-2:]
    else:
        left, right = next((pairs[i], pairs[i + 1]) for i in range(len(pairs) - 1) if pairs[i][0] <= value <= pairs[i + 1][0])
    if right[0] <= left[0]:
        return value + left[1] - left[0]
    return left[1] + (value - left[0]) * (right[1] - left[1]) / (right[0] - left[0])


def validate_markers(markers: dict) -> dict:
    result = {}
    for medium in ("video", "signal"):
        entries = markers.get(medium, {})
        cleaned = {}
        values = []
        for key in MARKER_NAMES:
            value = entries.get(key)
            if value in (None, ""):
                cleaned[key] = None
            else:
                number = float(value)
                if not 0 <= number <= 86400:
                    raise ValueError("Метка этапа должна быть в пределах суток")
                cleaned[key] = number
                values.append(number)
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError(f"Метки {medium} должны идти по порядку без совпадений")
        result[medium] = cleaned
    return result


def validate_turns(turns: list[dict]) -> list[dict]:
    cleaned = []
    for turn in turns:
        start, end = float(turn["start"]), float(turn["end"])
        if start < 0 or end <= start:
            raise ValueError("Конец реплики должен быть позже начала")
        speaker = turn.get("speaker", "respondent")
        if speaker not in {"interviewer", "respondent"}:
            raise ValueError("Неизвестный говорящий")
        acoustic = turn.get("acoustic") or {}
        cleaned.append({
            "start": start, "end": end, "speaker": speaker,
            "text": str(turn.get("text", ""))[:10000],
            "pause_seconds": turn.get("pause_seconds"),
            "acoustic": {key: acoustic.get(key) for key in ("rms", "pitch_hz", "spectral_centroid_hz")},
        })
    return sorted(cleaned, key=lambda item: item["start"])


class SessionStore:
    def __init__(self, root: Path = DATA):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, sid: str) -> Path:
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError("Некорректный идентификатор сессии")
        path = self.root / sid
        if not (path / "session.json").is_file():
            raise FileNotFoundError("Сессия не найдена")
        return path

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*/session.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                result.append({"id": meta["id"], "name": meta["name"], "created": meta["created"]})
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return result

    def create(self, name: str) -> dict:
        sid = uuid.uuid4().hex
        directory = self.root / sid
        directory.mkdir()
        meta = {
            "id": sid, "name": name[:120] or "Новая сессия",
            "created": datetime.now(timezone.utc).isoformat(),
            "video": None, "signal": None, "markers": validate_markers({}),
            "turns": [], "annotations": [], "jobs": {}, "video_analysis": None,
        }
        self._write(directory, meta)
        return meta

    def read(self, sid: str) -> dict:
        return json.loads((self.directory(sid) / "session.json").read_text(encoding="utf-8"))

    def _write(self, directory: Path, meta: dict) -> None:
        temporary = directory / "session.json.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "session.json")

    def update(self, sid: str, **changes) -> dict:
        with LOCK:
            directory = self.directory(sid)
            meta = self.read(sid)
            meta.update(changes)
            self._write(directory, meta)
            return meta

    def video_path(self, sid: str) -> Path | None:
        meta = self.read(sid)
        video = meta.get("video")
        if not video:
            return None
        path = self.directory(sid) / video["filename"]
        return path if path.is_file() else None

    def import_video(self, sid: str, source: Path) -> dict:
        if source.suffix.lower() not in {".mp4", ".mov", ".m4v", ".webm"}:
            raise ValueError("Поддерживаются MP4, MOV, M4V и WebM")
        directory = self.directory(sid)
        target = directory / f"video{source.suffix.lower()}"
        temporary = directory / (target.name + ".part")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        (directory / "video_features.json").unlink(missing_ok=True)
        return self.update(sid, video={"filename": target.name, "size": target.stat().st_size, "duration": None}, video_analysis=None)

    def import_bioradio(self, sid: str, source: Path) -> dict:
        directory = self.directory(sid)
        summary = import_csv(source, directory)
        clear_hrv_cache()
        return self.update(sid, signal=summary)

    def analyze_video(self, sid: str, progress) -> dict:
        video = self.video_path(sid)
        if video is None:
            raise ValueError("Сначала добавьте видео")
        directory = self.directory(sid)
        result = analyze_video(video, directory / "video_features.json", progress)
        return self.update(sid, video_analysis=result)

    def transcribe(self, sid: str, progress) -> dict:
        video = self.video_path(sid)
        if video is None:
            raise ValueError("Сначала добавьте видео")
        turns = transcribe_and_diarize(video, progress)
        return self.update(sid, turns=validate_turns(turns))

    def signals(self, sid: str, start: float, end: float) -> dict:
        meta = self.read(sid)
        if not meta.get("signal"):
            return {"time": [], "channels": {}}
        markers = meta["markers"]
        mapped_start = map_time(start, markers["video"], markers["signal"])
        mapped_end = map_time(end, markers["video"], markers["signal"])
        result = plot_data(self.directory(sid), mapped_start, mapped_end)
        result["time"] = [map_time(t, markers["signal"], markers["video"]) for t in result["time"]]
        return result

    def stats(self, sid: str, start: float, end: float) -> dict:
        meta = self.read(sid)
        markers = meta["markers"]
        left = map_time(start, markers["video"], markers["signal"])
        right = map_time(end, markers["video"], markers["signal"])
        return interval_stats(self.directory(sid), start, end, left, right, meta.get("turns", []))
