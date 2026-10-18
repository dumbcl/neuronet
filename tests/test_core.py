import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from neuronet_lab.analytics import _speech_stats
from neuronet_lab.bioradio import import_csv, load_arrays, plot_data
from neuronet_lab.media import _decode_audio, _pulse, _voice_vector
from neuronet_lab.session import map_time


class ImportTests(unittest.TestCase):
    def test_bioradio_decimal_comma_and_regular(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            malformed_export = base / "eu.csv"
            malformed_export.write_text(
                "Elapsed Time,ECG,RIP,GSR,Flow Rate temp,Heart Rate pulse,PPG pulse,SpO2 pulse,BioRadio Event,\n"
                "00:00:00,0,125,-0,500,184,250,-13,440,98,77,250,97,0,\n"
                "00:00:00.001,0,250,-0,250,184,500,-13,430,98,77,500,97,0,\n",
                encoding="utf-8",
            )
            result = import_csv(malformed_export, base / "eu")
            _, times, signals = load_arrays(base / "eu")
            self.assertEqual(result["format"], "decimal_comma")
            self.assertEqual(result["sample_rate_hz"], 1000)
            self.assertAlmostEqual(signals[0, 0], 0.125)
            self.assertAlmostEqual(signals[0, 1], -0.5)
            self.assertAlmostEqual(signals[0, 5], 77.25)
            self.assertEqual(len(times), 2)

            regular = base / "regular.csv"
            regular.write_text("Elapsed Time,ECG,PPG pulse,Heart Rate pulse\n00:00:00,0.2,81.5,72\n00:00:00.002,0.3,82.5,72\n", encoding="utf-8")
            result = import_csv(regular, base / "regular")
            self.assertEqual(result["format"], "regular")
            self.assertEqual(result["sample_rate_hz"], 500)
            self.assertEqual(plot_data(base / "regular", 0, 1)["channels"]["PPG"], [81.5, 82.5])

    def test_time_alignment_uses_multiple_anchors(self):
        source = {"baseline_start": 10, "dialog_start": 130, "dialog_end": 430, "post_end": None}
        target = {"baseline_start": 20, "dialog_start": 140, "dialog_end": 455, "post_end": None}
        self.assertEqual(map_time(70, source, target), 80)
        self.assertAlmostEqual(map_time(280, source, target), 297.5)
        self.assertAlmostEqual(map_time(297.5, target, source), 280)

    def test_speech_latency_and_rate(self):
        turns = [
            {"start": 1, "end": 3, "speaker": "interviewer", "text": "Как вас зовут?"},
            {"start": 4.5, "end": 7.5, "speaker": "respondent", "text": "Меня зовут Алексей"},
        ]
        result = _speech_stats(turns, 0, 10)
        self.assertEqual(result["response_latency_seconds"], 1.5)
        self.assertEqual(result["words_per_minute"], 60)
        self.assertEqual(result["turn_count"], 2)

    def test_video_pulse_and_voice_features(self):
        fps = 10
        seconds = np.arange(120) / fps
        green = 110 + 2 * np.sin(2 * np.pi * 1.2 * seconds)
        rgb = np.column_stack([np.full(120, 100), green, np.full(120, 90)]).tolist()
        bpm, quality, pulse = _pulse(rgb, fps)
        self.assertAlmostEqual(bpm, 70, delta=6)
        self.assertGreater(quality, 0.1)
        self.assertIsNotNone(pulse)
        tone = np.sin(2 * np.pi * 180 * np.arange(16000) / 16000).astype(np.float32)
        self.assertEqual(len(_voice_vector(tone, 16000)), 7)

    def test_audio_decoder_with_current_pyav(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            samples = (10000 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)).astype("<i2")
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(samples.tobytes())
            decoded = _decode_audio(path)
            self.assertEqual(len(decoded), 16000)
            self.assertGreater(float(np.max(np.abs(decoded))), 0.2)


if __name__ == "__main__":
    unittest.main()
