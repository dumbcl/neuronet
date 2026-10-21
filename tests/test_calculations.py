import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neuronet_lab.bioradio import import_csv, load_arrays
from neuronet_lab.media import _acoustic_features, _acoustic_frames
from neuronet_lab.session import SessionStore, validate_turns
from neuronet_lab.signal_processing import heart_stats, periodogram, respiration
from neuronet_lab.speech import speech_stats
from neuronet_lab.video_features import video_stats


class ImportRegressionTests(unittest.TestCase):
    header = "Elapsed Time,ECG,RIP,GSR,Flow Rate temp,Heart Rate pulse,PPG pulse,SpO2 pulse,BioRadio Event,\n"

    def test_integer_gsr_keeps_following_channels_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text(self.header +
                "00:01:04.766,0,0275430530309677,0,414534449577332,125,-13,4139060974121,100,80,6555252075195,97,0,\n")
            meta = import_csv(source, path / "parsed")
            _, _, values = load_arrays(path / "parsed")
            self.assertEqual(meta["malformed_rows"], 0)
            np.testing.assert_allclose(values[0], [0.0275430530309677, 0.414534449577332, 125, -13.4139060974121, 100, 80.6555252075195, 97, 0])

    def test_row_format_can_change_after_first_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text(self.header + "0,0.1,0.2,125,-13.4,100,80.5,97,0,\n" +
                              "0.001,0,125,-0,500,184,250,-13,440,98,77,250,97,0,\n")
            meta = import_csv(source, path / "parsed")
            _, times, values = load_arrays(path / "parsed")
            self.assertEqual(len(times), 2)
            self.assertEqual(meta["malformed_rows"], 0)
            self.assertEqual(values[1, 5], 77.25)

    def test_split_decimal_scientific_notation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text(self.header + "0,6,02427026024088E-05,-0,00100015953648835,184,297821044922,-13,4402265548706,98,77,0348663330078,97,0,\n")
            meta = import_csv(source, path / "parsed")
            _, _, values = load_arrays(path / "parsed")
            self.assertEqual(meta["malformed_rows"], 0)
            self.assertAlmostEqual(values[0, 0], 6.02427026024088e-5)
            self.assertAlmostEqual(values[0, 1], -.00100015953648835)
            self.assertEqual(values[0, 4], 98)

    def test_ambiguous_row_keeps_timestamp_and_explicit_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text(self.header + "0,1,2,3,4,5,72,80,97,0,\n" + "0.001,0.1,0.2,125,-13.4,100,80.5,97,0,\n")
            meta = import_csv(source, path / "parsed")
            _, times, values = load_arrays(path / "parsed")
            self.assertEqual(meta["malformed_rows"], 1)
            self.assertEqual(times.tolist(), [0, 0.001])
            self.assertTrue(np.isnan(values[0]).all())
            self.assertEqual(values[1, 5], 80.5)

    def test_pulseox_sensor_names_and_integer_gsr(self):
        header = "Elapsed Time,ECG,RIP,GSR,Flow Rate Sensor 1,Heart Rate PulseOx1,PPG PulseOx1,SpO2 PulseOx1,BioRadio Event,\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "margaryan.csv"
            source.write_text(header + "00:00:00,-0,0172061063349247,-0,0107475062832236,100000,-5,84263038635254,100,80,2795486450195,100,0,\n")
            meta = import_csv(source, path / "parsed")
            _, _, values = load_arrays(path / "parsed")
            self.assertEqual(meta["channels"], ["ECG", "RIP", "GSR", "Flow", "HeartRate", "PPG", "SpO2", "Event"])
            self.assertEqual(meta["malformed_rows"], 0)
            np.testing.assert_allclose(values[0], [-.0172061063349247, -.0107475062832236, 100000, -5.84263038635254, 100, 80.2795486450195, 100, 0])
            self.assertEqual(meta["source_file"], "margaryan.csv")
            self.assertIn("PPG PulseOx1", meta["source_headers"])

    def test_empty_event_and_invalid_device_values_keep_other_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text(self.header +
                "0,0,0275430530309677,-0,414534449577332,100000,-13,4402265548706,110,80,6555252075195,95,\n" +
                "0.001,0,0275430530309677,-0,414534449577332,100000,-13,4402265548706,511,80,6555252075195,127,0,\n")
            meta = import_csv(source, path / "parsed")
            _, _, values = load_arrays(path / "parsed")
            self.assertEqual(meta["malformed_rows"], 0)
            self.assertAlmostEqual(values[0, 0], .0275430530309677)
            self.assertTrue(np.isnan(values[0, 7]))
            self.assertAlmostEqual(values[1, 0], .0275430530309677)
            self.assertTrue(np.isnan(values[1, 4]))
            self.assertTrue(np.isnan(values[1, 6]))
            self.assertEqual(meta["invalid_device_values"], {"HeartRate": 1, "SpO2": 1})

    def test_failed_import_does_not_overwrite_existing_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text("Elapsed Time,ECG\n0,0.125\n")
            output = path / "parsed"
            import_csv(source, output)
            saved = {name: (output / name).read_bytes() for name in ("time.npy", "signals.npy", "signal_meta.json")}
            source.write_text(self.header + "0,1,2,3,4,5,72,80,97,0,\n")
            with self.assertRaisesRegex(ValueError, "Прежние данные"):
                import_csv(source, output)
            for name, contents in saved.items():
                self.assertEqual((output / name).read_bytes(), contents)

    def test_duplicate_sensor_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.csv"
            source.write_text("Elapsed Time,PPG PulseOx1,PPG PulseOx2\n0,80.5,81.5\n")
            with self.assertRaisesRegex(ValueError, "несколько датчиков"):
                import_csv(source, path / "parsed")

    def test_semicolon_and_quoted_decimal_values(self):
        for content in ('Elapsed Time;ECG;PPG pulse\n0;0,125;80,5\n', 'Elapsed Time,ECG,PPG pulse\n0,"0,125","80,5"\n'):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                source = path / "input.csv"
                source.write_text(content)
                import_csv(source, path / "parsed")
                _, _, values = load_arrays(path / "parsed")
                np.testing.assert_allclose(values[0], [0.125, 80.5])


class SpeechRegressionTests(unittest.TestCase):
    def setUp(self):
        self.turns = [dict(start=1, end=2, speaker="interviewer", text="Первый фрагмент"),
                      dict(start=2.1, end=3, speaker="interviewer", text="вопроса?"),
                      dict(start=4, end=5, speaker="respondent", text="Первый ответ", pause_seconds=0),
                      dict(start=6, end=8, speaker="respondent", text="Продолжение ответа", pause_seconds=0)]

    def test_latency_uses_context_and_final_question_fragment_once(self):
        result = speech_stats(self.turns, 6, 8)
        self.assertEqual(result["response_latency_seconds"], 1)
        self.assertEqual(result["latency_count"], 1)
        self.assertEqual(result["response_duration_seconds"], 4)

    def test_answer_duration_includes_between_fragment_pauses(self):
        result = speech_stats(self.turns, 4, 8)
        self.assertEqual(result["internal_pause_seconds"], 1)
        self.assertEqual(result["words_per_minute"], 60)

    def test_partial_turn_does_not_count_entire_text(self):
        result = speech_stats(self.turns, 6.5, 7)
        self.assertIsNone(result["respondent_words"])
        self.assertIsNone(result["words_per_minute"])
        self.turns[-1]["words"] = [dict(start=6.1, end=6.4, word="Продолжение"), dict(start=6.6, end=6.9, word="ответа")]
        result = speech_stats(self.turns, 6.5, 7)
        self.assertEqual(result["respondent_words"], 1)

    def test_full_audio_rms_and_voicing(self):
        fs = 16000
        t = np.arange(fs) / fs
        audio = np.concatenate([0.01 * np.sin(2 * np.pi * 180 * t)] * 4 + [0.1 * np.sin(2 * np.pi * 240 * t)] * 4)
        result = _acoustic_features(audio, fs, np.zeros(7))
        self.assertAlmostEqual(result["rms"], float(np.sqrt(np.mean(audio ** 2))), places=12)
        self.assertGreater(result["pitch_iqr_hz"], 45)
        silence = _acoustic_features(np.zeros(fs), fs, np.zeros(7))
        self.assertIsNone(silence["pitch_hz"])
        noise = _acoustic_frames(np.random.default_rng(1).normal(0, .01, fs), fs)
        self.assertEqual(np.isfinite(noise["pitch_hz"]).sum(), 0)
        low = _acoustic_features(.01 * np.sin(2 * np.pi * 70 * t), fs, np.zeros(7))
        self.assertAlmostEqual(low["pitch_hz"], 70, delta=1)

    def test_acoustics_do_not_average_turns_with_equal_weights(self):
        turns = [dict(start=0, end=1, speaker="respondent", text="Один", acoustic=dict(version=2, rms=.01)),
                 dict(start=1, end=10, speaker="respondent", text="Два", acoustic=dict(version=2, rms=.1))]
        self.assertAlmostEqual(speech_stats(turns, 0, 10)["voice_rms"], np.sqrt((.01 ** 2 + 9 * .1 ** 2) / 10))
        cleaned = validate_turns(turns)
        self.assertEqual(cleaned[0]["acoustic"]["version"], 2)

    def test_negative_autocorrelation_peaks_leave_pitch_unavailable(self):
        fs = 16000
        t = np.arange(fs) / fs
        # Low-frequency rumble plus noise produces frames whose local maxima
        # in the pitch search band are all negative (as in Margaryan's audio).
        audio = .01 * (np.sin(2 * np.pi * 30 * t) + .2 * np.random.default_rng(0).normal(size=len(t)))
        frames = _acoustic_frames(audio, fs)
        self.assertEqual(len(frames["start"]), 48)
        self.assertTrue(np.isfinite(frames["rms"]).all())
        self.assertTrue(np.isfinite(frames["spectral_centroid_hz"]).all())
        self.assertTrue(np.isnan(frames["pitch_hz"][26]))
        isolated = _acoustic_frames(audio[8320:9120], fs)
        self.assertTrue(np.isnan(isolated["pitch_hz"][0]))

    def test_reprocessing_preserves_manual_session_and_saves_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            meta = store.create("Проверка")
            sid = meta["id"]
            (store.directory(sid) / "video.mp4").touch()
            meta = store.update(sid, video={"filename": "video.mp4"}, turns=self.turns,
                                annotations=[dict(start=4, end=8, label="truth", name="Ответ")])
            fs = 16000
            audio = .01 * np.sin(2 * np.pi * 180 * np.arange(10 * fs) / fs)
            with patch("neuronet_lab.media._decode_audio", return_value=audio):
                result = store.refresh_acoustics(sid, lambda value: None)
            self.assertEqual(result["markers"], meta["markers"])
            self.assertEqual(result["annotations"], meta["annotations"])
            for original, updated in zip(self.turns, result["turns"]):
                for key in ("text", "start", "end", "speaker"):
                    self.assertEqual(original[key], updated[key])
                self.assertEqual(updated["acoustic"]["version"], 2)
            stats = store.stats(sid, 6.2, 7.8)
            self.assertAlmostEqual(stats["speech"]["pitch_hz"], 180, delta=1)
            self.assertGreater(stats["speech"]["acoustic_coverage"], .95)
            self.assertIsNone(stats["speech"]["words_per_minute"])


class AnswerDatasetTests(unittest.TestCase):
    def setUp(self):
        self.turns = [dict(id="q", start=0, end=1, speaker="interviewer", text="Вопрос?"),
                      dict(id="r1", start=2, end=4, speaker="respondent", text="Первое предложение", pause_seconds=0),
                      dict(id="r2", start=7, end=9, speaker="respondent", text="Второе предложение", pause_seconds=0)]
        self.answers = [dict(id="a", start=2, end=9, label="truth", turn_ids=["r1", "r2"],
                             question_turn_ids=["q"], internal_pauses=[dict(start=4, end=7)])]

    def test_full_answer_includes_pause_without_double_counting(self):
        stats = speech_stats(self.turns, 2, 9, answer_spans=self.answers)
        self.assertEqual(stats["answer_count"], 1)
        self.assertEqual(stats["answer_ids"], ["a"])
        self.assertEqual(stats["response_duration_seconds"], 7)
        self.assertEqual(stats["internal_pause_seconds"], 3)
        self.assertAlmostEqual(stats["words_per_minute"], 4 * 60 / 7)
        self.assertEqual(stats["response_latency_seconds"], 1)
        pause = speech_stats(self.turns, 4.5, 6.5, answer_spans=self.answers)
        self.assertEqual(pause["answer_ids"], ["a"])
        self.assertEqual(pause["internal_pause_seconds"], 2)
        self.assertEqual(pause["response_duration_seconds"], 7)

    def test_internal_pause_inside_single_turn(self):
        turns = [dict(id="r", start=2, end=12, speaker="respondent", text="Ответ", pause_seconds=4)]
        answers = [dict(id="a", turn_ids=["r"], internal_pauses=[dict(start=5, end=8)])]
        stats = speech_stats(turns, 2, 12, answer_spans=answers)
        self.assertEqual(stats["internal_pause_seconds"], 4)

    def test_ids_survive_save_and_changed_labels_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            sid = store.create("Dataset")['id']
            annotations = [dict(start=2, end=9, label="truth")]
            store.update(sid, turns=self.turns, answers=self.answers, annotations=annotations, answer_review_required=False)
            saved = store.update(sid, turns=validate_turns(self.turns))
            self.assertFalse(saved["answer_review_required"])
            self.assertEqual(saved["turns"][1]["id"], "r1")
            changed = store.update(sid, annotations=[dict(start=2, end=9, label="lie")])
            self.assertTrue(changed["answer_review_required"])
            store.update(sid, answers=self.answers, answer_review_required=False)
            turns = [dict(t) for t in self.turns]
            turns[1]["end"] = 4.5
            changed = store.update(sid, turns=turns)
            self.assertTrue(changed["answer_review_required"])


class PhysiologyRegressionTests(unittest.TestCase):
    def test_psd_power_has_physical_scaling_independent_of_duration(self):
        powers = []
        for seconds in (120, 240):
            t = np.arange(seconds * 4) / 4
            hz, density = periodogram(100 * np.sin(2 * np.pi * .1 * t), 4)
            powers.append(float(density.sum() * (hz[1] - hz[0])))
        np.testing.assert_allclose(powers, [5000, 5000], rtol=.001)

    def test_gap_prevents_hrv_and_is_not_compressed_away(self):
        fs = 100
        t = np.arange(121 * fs) / fs
        peaks = np.arange(fs, 120 * fs, fs)
        t[6000:] += 10
        with patch("neuronet_lab.signal_processing.pulse_peaks", return_value=peaks):
            result = heart_stats(np.ones(len(t)), fs, "ecg", t[-1], t)
        self.assertEqual(result["heart_rate_bpm"], 60)
        self.assertIsNone(result["lf_power"])
        self.assertIsNone(result["rmssd_ms"])
        self.assertGreater(result["rejected_rr_count"], 0)

    def test_rr_spectrum_and_short_interval_gate(self):
        fs = 1000
        beat = [0.5]
        while beat[-1] < 241:
            beat.append(beat[-1] + .8 + .05 * np.sin(2 * np.pi * .1 * beat[-1]))
        peaks = np.asarray(np.array(beat) * fs, dtype=int)
        x = np.zeros(242 * fs)
        with patch("neuronet_lab.signal_processing.pulse_peaks", return_value=peaks):
            result = heart_stats(x, fs, "ecg", 242)
        self.assertEqual(result["quality"], "ok")
        self.assertGreater(result["lf_power"], result["hf_power"] * 10)
        self.assertGreater(result["lf_power"], 500)
        self.assertIsNone(result["vlf_power"])
        self.assertIsNone(result["ulf_power"])
        short = heart_stats(np.sin(np.arange(1000)), fs, "ppg", 1)
        self.assertIsNone(short["sdnn_ms"])

    def test_respiration_keeps_time_axis(self):
        fs = 100
        t = np.arange(60 * fs) / fs
        self.assertAlmostEqual(respiration(np.sin(2 * np.pi * .25 * t), fs, t), 15)
        t[3000:] += 5
        self.assertIsNone(respiration(np.sin(2 * np.pi * .25 * t), fs, t))

    def test_synthetic_ecg_and_ppg_rates(self):
        fs = 200
        t = np.arange(fs * 125) / fs
        ppg = np.sin(2 * np.pi * 1.25 * t)
        self.assertAlmostEqual(heart_stats(ppg, fs, "ppg", 125)["heart_rate_bpm"], 75, delta=1)
        ecg = np.zeros(len(t))
        for beat in np.arange(.5, 124, .8):
            ecg += np.exp(-((t - beat) / .012) ** 2)
        self.assertAlmostEqual(heart_stats(ecg, fs, "ecg", 125)["heart_rate_bpm"], 75, delta=1)


class VideoWindowTests(unittest.TestCase):
    def test_legacy_windows_reset_on_face_loss_and_head_speed_has_time_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            points = [dict(time=i / 10, rppg_bpm=75 if i >= 80 else None, rppg_quality=.3 if i >= 80 else None,
                           face_bbox=[0, 0, .2, .2] if i != 100 else None, head_motion=.01) for i in range(250)]
            (path / "video_features.json").write_text(json.dumps(points))
            stats = video_stats(path, 10.1, 25)
            self.assertAlmostEqual(stats["head_motion"], .1)
            self.assertAlmostEqual(stats["rppg_window_start"], 10.1)

    def test_short_interval_excludes_rppg_history_but_keeps_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            points = [dict(time=float(t), feature_version=2, rppg_bpm=75, rppg_quality=.3, rppg_window_start=float(t - 12), rppg_window_end=float(t)) for t in range(12, 26)]
            (path / "video_features.json").write_text(json.dumps(points))
            short = video_stats(path, 20, 22)
            self.assertIsNone(short["rppg_heart_rate_bpm"])
            self.assertEqual(short["rppg_context_heart_rate_bpm"], 75)
            long = video_stats(path, 0, 26)
            self.assertEqual(long["rppg_heart_rate_bpm"], 75)
            self.assertEqual(long["rppg_window_count"], 14)


if __name__ == "__main__":
    unittest.main()
