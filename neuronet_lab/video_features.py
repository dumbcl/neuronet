"""Window-aware summaries, including conservative support for legacy caches."""
import json
from functools import lru_cache

import numpy as np


def _mean(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


@lru_cache(maxsize=4)
def load_video_points(path, modified_ns):
    points = json.loads(path.read_text(encoding="utf-8"))
    differences = np.diff([p["time"] for p in points])
    step = float(np.median(differences[differences > 0])) if np.any(differences > 0) else 0.1
    face_start, previous = None, None
    for p in points:
        if p.get("face_bbox") is None:
            face_start = None
        elif face_start is None:
            face_start = p["time"]
        if p.get("feature_version", 1) < 2:
            # Legacy algorithm: at most floor(fps*12) samples, reset on lost face.
            if p.get("rppg_quality") is not None and face_start is not None:
                p["rppg_window_start"] = max(face_start, p["time"] - (max(1, int(12 / step + 1e-6)) - 1) * step)
                p["rppg_window_end"] = p["time"]
            if p.get("head_motion") is not None and previous is not None:
                dt = p["time"] - previous
                p["head_motion"] = p["head_motion"] / dt if dt > 0 else None
        previous = p["time"]
    return points


def video_stats(session_dir, start, end):
    source = session_dir / "video_features.json"
    empty = dict(rppg_heart_rate_bpm=None, rppg_context_heart_rate_bpm=None,
                 head_motion=None, hand_motion=None, leg_motion=None, trunk_tilt_degrees=None,
                 quality="not_processed", rppg_coverage=0.0, rppg_window_count=0)
    if not source.exists():
        return empty
    points = load_video_points(source, source.stat().st_mtime_ns)
    selected = [p for p in points if start <= p["time"] < end]
    valid = [p for p in selected if p.get("rppg_bpm") is not None and
             p.get("rppg_quality") is not None and p["rppg_quality"] >= 0.12 and
             p.get("rppg_window_start") is not None and p.get("rppg_window_end") is not None and
             p["rppg_window_start"] >= start - 1e-6 and p["rppg_window_end"] <= end + 1e-6]
    return dict(rppg_heart_rate_bpm=_mean([p["rppg_bpm"] for p in valid]),
                rppg_context_heart_rate_bpm=_mean([p.get("rppg_bpm") for p in selected]),
                rppg_coverage=len(valid) / len(selected) if selected else 0.0,
                rppg_window_count=len(valid),
                rppg_window_start=min(p["rppg_window_start"] for p in valid) if valid else None,
                rppg_window_end=max(p["rppg_window_end"] for p in valid) if valid else None,
                head_motion=_mean([p.get("head_motion") for p in selected]),
                hand_motion=_mean([p.get("hand_motion") for p in selected]),
                leg_motion=_mean([p.get("leg_motion") for p in selected]),
                trunk_tilt_degrees=_mean([p.get("trunk_tilt") for p in selected]),
                motion_units="head: face_width/s; limbs: image_height/s" if selected and all(p.get("feature_version", 1) >= 2 for p in selected) else "head: face_width/s; limbs: legacy image coordinates/s",
                legacy_pose=any(p.get("feature_version", 1) < 2 for p in selected),
                quality="window_supported" if valid else "insufficient_window_or_face_signal",
                landmark_coverage={key: sum(p.get(key) is not None for p in selected) / len(selected) if selected else 0.0 for key in ("head_motion", "hand_motion", "leg_motion", "trunk_tilt")})
