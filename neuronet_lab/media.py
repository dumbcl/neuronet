"""Optional offline video and speech extraction.

The viewer and BioRadio analysis work without these extra packages. Media
jobs report a concrete error if OpenCV or faster-whisper is not installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _pulse(rgb: list[list[float]], fps: float) -> tuple[float | None, float | None, float | None]:
    """POS projection of normalized RGB followed by a physiological band FFT."""
    if len(rgb) < max(60, int(fps * 8)):
        return None, None, None
    x = np.asarray(rgb[-int(fps * 12) :], dtype=float)
    if not np.isfinite(x).all() or np.min(np.mean(x, axis=0)) < 1:
        return None, None, None
    x = x / np.mean(x, axis=0) - 1
    s1 = x[:, 1] - x[:, 2]
    s2 = x[:, 1] + x[:, 2] - 2 * x[:, 0]
    signal = s1 + np.std(s1) / max(np.std(s2), 1e-8) * s2
    signal -= np.mean(signal)
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / fps)
    band = (freqs >= 0.7) & (freqs <= 3.2)
    if not band.any() or np.sum(power[band]) <= 0:
        return None, None, None
    peak_index = np.flatnonzero(band)[np.argmax(power[band])]
    quality = power[peak_index] / np.sum(power[band])
    filtered = np.fft.rfft(signal)
    filtered[~band] = 0
    pulse_value = float(np.fft.irfft(filtered, n=len(signal))[-1])
    if quality < 0.12:
        return None, float(quality), None
    return float(freqs[peak_index] * 60), float(quality), pulse_value


def analyze_video(video_path: Path, output_path: Path, progress) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Для анализа видео установите opencv-python из requirements-video.txt") from exc
    bundled_detector = Path(__file__).resolve().parents[1] / "models" / "haarcascade_frontalface_default.xml"
    detector_path = bundled_detector if bundled_detector.is_file() else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(detector_path))
    if face_detector.empty():
        raise RuntimeError("OpenCV could not load its frontal-face detector")
    pose = None
    mp = None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_heavy.task"
        if model_path.exists():
            options = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model_path)))
            pose = vision.PoseLandmarker.create_from_options(options)
    except (ImportError, AttributeError):
        pass
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video could not be opened by OpenCV")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or not count:
        cap.release()
        raise RuntimeError("Video FPS or frame count is unavailable")
    stride = max(1, round(fps / 10))
    actual_fps = fps / stride
    points = []
    rgb_history: list[list[float]] = []
    last_face = None
    last_joints = {}
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number % stride:
            frame_number += 1
            continue
        t = frame_number / fps
        small = cv2.resize(frame, (min(frame.shape[1], 640), round(frame.shape[0] * min(frame.shape[1], 640) / frame.shape[1])))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        bpm = quality = pulse_value = motion = tilt = face_bbox = pose_landmarks = hand_motion = leg_motion = None
        rppg_window_start = rppg_window_end = None
        if len(faces):
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            # Central forehead and both cheeks; skip eyes and the mouth.
            regions = [
                small[y + int(h * 0.12) : y + int(h * 0.3), x + int(w * 0.25) : x + int(w * 0.75)],
                small[y + int(h * 0.5) : y + int(h * 0.72), x + int(w * 0.08) : x + int(w * 0.35)],
                small[y + int(h * 0.5) : y + int(h * 0.72), x + int(w * 0.65) : x + int(w * 0.92)],
            ]
            pixels = np.concatenate([r.reshape(-1, 3) for r in regions if r.size])
            rgb_history.append(np.mean(pixels[:, ::-1], axis=0).tolist())
            rgb_history = rgb_history[-int(actual_fps * 12):]
            bpm, quality, pulse_value = _pulse(rgb_history, actual_fps)
            if quality is not None:
                rppg_window_start, rppg_window_end = t - (len(rgb_history) - 1) / actual_fps, t
            face_bbox = [round(x / small.shape[1], 4), round(y / small.shape[0], 4), round(w / small.shape[1], 4), round(h / small.shape[0], 4)]
            center = (x + w / 2, y + h / 2)
            if last_face:
                motion = float(math.dist(center, last_face) / max(w, 1) * actual_fps)
            last_face = center
        else:
            rgb_history.clear()
            last_face = None
        if pose:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            result = pose.detect(mp_image)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                pose_landmarks = [[round(landmarks[i].x, 4), round(landmarks[i].y, 4)] for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28) if landmarks[i].visibility > 0.5]
                try:
                    shoulder = np.mean([[landmarks[i].x, landmarks[i].y] for i in (11, 12)], axis=0)
                    hip = np.mean([[landmarks[i].x, landmarks[i].y] for i in (23, 24)], axis=0)
                    if all(landmarks[i].visibility > 0.5 for i in (11, 12, 23, 24)):
                        dx, dy = (shoulder - hip) * [small.shape[1] / small.shape[0], 1]
                        tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
                except (IndexError, AttributeError):
                    pass
                current_joints = {i: np.array([landmarks[i].x * small.shape[1] / small.shape[0], landmarks[i].y]) for i in (15, 16, 27, 28) if landmarks[i].visibility > 0.5}
                hand_speeds = [float(np.linalg.norm(current_joints[i] - last_joints[i]) * actual_fps) for i in (15, 16) if i in current_joints and i in last_joints]
                leg_speeds = [float(np.linalg.norm(current_joints[i] - last_joints[i]) * actual_fps) for i in (27, 28) if i in current_joints and i in last_joints]
                hand_motion = float(np.mean(hand_speeds)) if hand_speeds else None
                leg_motion = float(np.mean(leg_speeds)) if leg_speeds else None
                last_joints = current_joints
            else:
                last_joints = {}
        points.append({"time": round(t, 3), "rppg_bpm": bpm, "rppg_quality": quality, "rppg_value": pulse_value, "rppg_window_start": rppg_window_start, "rppg_window_end": rppg_window_end, "feature_version": 2, "motion_units": "head: face_width/s; limbs: image_height/s", "head_motion": motion, "hand_motion": hand_motion, "leg_motion": leg_motion, "trunk_tilt": tilt, "face_bbox": face_bbox, "pose_landmarks": pose_landmarks})
        frame_number += 1
        if len(points) % 100 == 0:
            progress(min(99, round(frame_number / count * 100)))
    cap.release()
    if pose:
        pose.close()
    output_path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
    progress(100)
    return {"duration": count / fps, "frames_analyzed": len(points), "pose_available": pose is not None, "rppg_points": sum(p["rppg_bpm"] is not None for p in points), "feature_version": 2}


def _voice_vector(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio) < sample_rate // 3:
        return np.zeros(7)
    audio = audio[: min(len(audio), 4 * sample_rate)]
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freq = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    bands = [(80, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    features = [np.log1p(float(np.sqrt(np.mean(audio * audio))))]
    features += [np.log1p(float(np.mean(spectrum[(freq >= lo) & (freq < hi)]))) for lo, hi in bands]
    # Pitch from autocorrelation on a short voiced interval.
    clip = audio[: min(len(audio), sample_rate)]
    decimated = clip[::4]
    fft_size = 1 << (2 * len(decimated) - 1).bit_length()
    spectrum_for_corr = np.fft.rfft(decimated, n=fft_size)
    corr = np.fft.irfft(np.abs(spectrum_for_corr) ** 2, n=fft_size)[: len(decimated)]
    low, high = max(1, int(sample_rate / 4 / 400)), int(sample_rate / 4 / 70)
    pitch = sample_rate / 4 / (low + np.argmax(corr[low:high])) if high > low else 0
    features.append(pitch / 200)
    return np.asarray(features)


def _acoustic_features(audio: np.ndarray, sample_rate: int, vector: np.ndarray) -> dict:
    # Keep the vector parameter for callers; diarization features are not F0 measurements.
    frames = _acoustic_frames(audio, sample_rate)
    return _summarize_acoustics(audio, frames, sample_rate)


def _acoustic_frames(audio: np.ndarray, sample_rate: int) -> dict:
    """50 ms windows / 20 ms hop, voiced autocorrelation F0 (70–400 Hz).

    F0 uses local autocorrelation peaks with periodicity and energy gates.
    This is a conservative estimator, not a validated speaker or emotion model.
    """
    x = np.asarray(audio, dtype=np.float64)
    width, hop = round(sample_rate * 0.05), round(sample_rate * 0.02)
    empty = {key: np.array([], dtype=float) for key in ("start", "end", "rms", "pitch_hz", "spectral_centroid_hz")}
    if len(x) < width:
        return empty
    starts = np.arange(0, len(x) - width + 1, hop)
    result = dict(start=starts / sample_rate, end=(starts + width) / sample_rate,
                  rms=np.zeros(len(starts)), pitch_hz=np.full(len(starts), np.nan),
                  spectral_centroid_hz=np.zeros(len(starts)))
    fft_size = 1 << (2 * width - 1).bit_length()
    freq = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    weights = np.hanning(width)
    window_corr = np.fft.irfft(np.abs(np.fft.rfft(weights, n=fft_size)) ** 2, n=fft_size)[:width]
    lo, hi = int(np.floor(sample_rate / 400)), min(width - 2, int(np.ceil(sample_rate / 70)) + 1)
    for first in range(0, len(starts), 256):
        indices = starts[first:first + 256, None] + np.arange(width)
        frames = x[indices]
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frames = (frames - frames.mean(axis=1, keepdims=True)) * weights
        spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
        magnitude = np.abs(spectrum)
        centroid = (magnitude * freq).sum(axis=1) / np.maximum(magnitude.sum(axis=1), 1e-20)
        correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size, axis=1)[:, :width]
        # Correct the lag-dependent attenuation of the Hann window, important
        # for low male F0 near the lower end of the supported range.
        correlation = correlation / np.maximum(correlation[:, :1], 1e-20)
        correlation *= window_corr[0] / np.maximum(window_corr, 1e-20)
        for row, corr in enumerate(correlation):
            if corr[0] <= 1e-16:
                continue
            peaks = np.flatnonzero((corr[lo:hi] > corr[lo - 1:hi - 1]) & (corr[lo:hi] >= corr[lo + 1:hi + 1])) + lo
            if not len(peaks):
                continue
            best = corr[peaks].max()
            # Prefer the fundamental over autocorrelation peaks at multiples of its period.
            lag = int(peaks[np.flatnonzero(corr[peaks] >= 0.9 * best)[0]])
            if corr[lag] / corr[0] < 0.6:
                continue
            denominator = corr[lag - 1] - 2 * corr[lag] + corr[lag + 1]
            offset = 0.5 * (corr[lag - 1] - corr[lag + 1]) / denominator if denominator else 0
            estimate = sample_rate / (lag + np.clip(offset, -0.5, 0.5))
            if 69.5 <= estimate <= 400.5:
                result["pitch_hz"][first + row] = float(np.clip(estimate, 70, 400))
        result["rms"][first:first + len(rms)] = rms
        result["spectral_centroid_hz"][first:first + len(rms)] = centroid
    gate = max(1e-5, float(np.percentile(result["rms"], 95)) * 0.03)
    result["pitch_hz"][result["rms"] < gate] = np.nan
    return result


def _summarize_acoustics(audio, frames, sample_rate=16000):
    pitch = frames["pitch_hz"][np.isfinite(frames["pitch_hz"])]
    return dict(version=2, scope="full_turn", duration_seconds=len(audio) / sample_rate,
                rms=float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))) if len(audio) else None,
                pitch_hz=float(np.median(pitch)) if len(pitch) else None,
                pitch_iqr_hz=float(np.percentile(pitch, 75) - np.percentile(pitch, 25)) if len(pitch) else None,
                voiced_fraction=len(pitch) / len(frames["start"]) if len(frames["start"]) else 0,
                spectral_centroid_hz=float(np.mean(frames["spectral_centroid_hz"])) if len(frames["start"]) else None)


def refresh_acoustics(video_path: Path, turns: list[dict], output_path: Path, progress) -> list[dict]:
    """Recompute audio without ASR or changing manual text, roles and annotations."""
    audio = _decode_audio(video_path, sampling_rate=16000)
    progress(5)
    collected, updated = [], []
    for i, turn in enumerate(turns):
        left, right = int(turn["start"] * 16000), int(turn["end"] * 16000)
        clip = audio[left:right]
        frames = _acoustic_frames(clip, 16000)
        updated.append(dict(turn, acoustic=_summarize_acoustics(clip, frames)))
        frames["start"] += left / 16000
        frames["end"] += left / 16000
        collected.append(frames)
        progress(min(99, 5 + int(94 * (i + 1) / max(1, len(turns)))))
    keys = ("start", "end", "rms", "pitch_hz", "spectral_centroid_hz")
    arrays = {key: np.concatenate([frame[key] for frame in collected]) if collected else np.array([]) for key in keys}
    order = np.argsort(arrays["start"], kind="stable")
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: values[order] for key, values in arrays.items()})
    temporary.replace(output_path)
    progress(100)
    return updated


def _two_speaker_labels(vectors: list[np.ndarray]) -> list[str]:
    if len(vectors) < 3:
        return ["respondent"] * len(vectors)
    x = np.asarray(vectors)
    std = np.std(x, axis=0)
    x = (x - np.mean(x, axis=0)) / np.where(std > 1e-6, std, 1)
    centers = x[[0, len(x) // 2]].copy()
    for _ in range(20):
        labels = np.argmin(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        for cluster in (0, 1):
            if np.any(labels == cluster):
                centers[cluster] = np.mean(x[labels == cluster], axis=0)
    first_cluster = int(labels[0])
    return ["interviewer" if label == first_cluster else "respondent" for label in labels]


def _decode_audio(video_path: Path, sampling_rate: int = 16000) -> np.ndarray:
    """Decode through PyAV without faster-whisper's version-specific open options."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Для расшифровки установите PyAV из requirements-audio.txt") from exc
    chunks = []
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("В видео нет звуковой дорожки для расшифровки")
        for frame in container.decode(container.streams.audio[0]):
            chunks.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(frame))
        chunks.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(None))
    if not chunks:
        raise ValueError("Не удалось декодировать звук из видео")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_and_diarize(video_path: Path, progress) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Для расшифровки установите faster-whisper из requirements-audio.txt") from exc
    progress(2)
    audio = _decode_audio(video_path, sampling_rate=16000)
    progress(5)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language="ru", vad_filter=True, word_timestamps=True)
    items = []
    vectors = []
    for segment in segments:
        start, end = float(segment.start), float(segment.end)
        clip = audio[int(start * 16000) : int(end * 16000)]
        vector = _voice_vector(clip, 16000)
        words = list(segment.words or [])
        pause_seconds = sum(max(0, words[i].start - words[i - 1].end) for i in range(1, len(words)) if words[i].start - words[i - 1].end >= 0.25)
        items.append({"start": round(start, 3), "end": round(end, 3), "text": segment.text.strip(), "pause_seconds": float(pause_seconds), "words": [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in words]})
        vectors.append(vector)
        progress(min(95, 5 + len(items)))
    labels = _two_speaker_labels(vectors)
    for item, label in zip(items, labels):
        item["speaker"] = label
    progress(100)
    return items
