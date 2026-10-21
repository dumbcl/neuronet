import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neuronet_lab.dataset import COLUMNS, dataset_filename
from neuronet_lab.episodes import answer_episode
from neuronet_lab.session import SessionStore, validate_markers


def fixture(root):
    store = SessionStore(root)
    sid = store.create("Участник")['id']
    directory = store.directory(sid)
    turns = [dict(id="q", start=0, end=1, speaker="interviewer", text='Вопрос, "пример"\nпродолжение?'),
             dict(id="r1", start=2, end=4, speaker="respondent", text="Первое предложение", pause_seconds=0),
             dict(id="r2", start=7, end=9, speaker="respondent", text="Второе предложение", pause_seconds=0),
             dict(id="q2", start=9.5, end=10.5, speaker="interviewer", text="Другой вопрос?"),
             dict(id="r3", start=11, end=12, speaker="respondent", text="Нет", pause_seconds=0),
             dict(id="other", start=3, end=3.5, speaker="respondent", text="Не включать в этот ответ")]
    answers = [dict(id="a1", participant_id="participant", start=2, end=9, label="truth", dataset_eligible=True,
                    turn_ids=["r1", "r2"], question_turn_ids=["q"], question_start=0, question_end=1,
                    internal_pauses=[dict(start=4, end=7)], source_annotation_ids=["src1"]),
               dict(id="a2", participant_id="participant", start=11, end=12, label="lie", dataset_eligible=True,
                    turn_ids=["r3"], question_turn_ids=["q2"], question_start=9.5, question_end=10.5),
               dict(id="base", start=12, end=14, label="base", dataset_eligible=False, turn_ids=[])]
    markers = validate_markers(dict(video=dict(baseline_start=0), signal=dict(baseline_start=100)))
    fs = 100
    times = 100 + np.arange(1401) / fs
    values = np.column_stack([np.sin(2*np.pi*1.2*times), np.where(times < 110, 72, 85),
                              np.full(len(times), 100000)])
    signal = dict(channels=["PPG", "HeartRate", "GSR"], sample_rate_hz=fs, sample_count=len(times),
                  start=100, end=114, import_version=4)
    (directory / "signal_meta.json").write_text(json.dumps(signal))
    np.save(directory / "signals.npy", values)
    np.save(directory / "time.npy", times)
    np.savez(directory / "audio_features.npz", start=[2.1, 7.1, 11.1], end=[3.9, 8.9, 11.9],
             rms=[.003271234567, .003271234567, .01], pitch_hz=[180, 180, 150], spectral_centroid_hz=[700, 700, 800])
    store.update(sid, turns=turns, answers=answers, markers=markers, signal=signal, answer_review_required=False)
    return store, sid


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class DatasetTests(unittest.TestCase):
    def test_export_targets_pauses_alignment_precision_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            before = store.read(sid)
            progress = []
            result = store.export_dataset(sid, "Исследование.csv", progress.append)
            self.assertEqual(store.read(sid), before)
            self.assertEqual(progress[-1], 100)
            self.assertEqual(Path(result['csv_path']).parent, store.directory(sid))
            rows = read_rows(result['csv_path'])
            self.assertEqual(tuple(rows[0]), COLUMNS)
            self.assertEqual([r['target'] for r in rows], ['0', '1'])
            self.assertEqual(rows[0]['answer_duration_seconds'], '7')
            self.assertEqual(rows[0]['speech_respondent_words'], '4')
            self.assertEqual(float(rows[0]['speech_internal_pause_seconds']), 3)
            self.assertAlmostEqual(float(rows[0]['speech_words_per_minute']), 240 / 7)
            self.assertAlmostEqual(float(rows[0]['speech_voice_rms']), .003271234567)
            self.assertEqual(float(rows[0]['signal_start_seconds']), 102)
            self.assertEqual(float(rows[0]['contact_heart_rate_monitor_bpm']), 72)
            self.assertEqual(float(rows[1]['contact_heart_rate_monitor_bpm']), 85)
            self.assertEqual(rows[0]['contact_gsr_mean'], '')
            self.assertEqual(float(rows[0]['signal_gsr_valid_coverage']), 0)
            self.assertEqual(rows[0]['contact_ppg_sdnn_ms'], '')
            self.assertEqual(rows[0]['question_text'], 'Вопрос, "пример"\nпродолжение?')
            self.assertEqual(rows[0]['dataset_schema_version'], '2')
            self.assertEqual(float(rows[0]['episode_start_seconds']), 0)
            self.assertEqual(float(rows[0]['episode_duration_seconds']), 9)
            self.assertEqual(float(rows[0]['question_duration_seconds']), 1)
            self.assertEqual(float(rows[0]['waiting_duration_seconds']), 1)
            self.assertEqual(float(rows[0]['question_signal_start_seconds']), 100)
            self.assertEqual(float(rows[0]['waiting_signal_start_seconds']), 101)
            self.assertEqual(float(rows[0]['delta_answer_minus_question_contact_heart_rate_monitor_bpm']), 0)
            self.assertNotIn('episode_speech_voice_rms', rows[0])
            meta = json.loads(Path(result['metadata_path']).read_text())
            self.assertEqual(meta['target_mapping'], dict(truth=0, lie=1))
            self.assertEqual(meta['row_count'], 2)
            self.assertEqual(meta['excluded_answers'][0]['answer_id'], 'base')
            self.assertEqual(meta['answer_snapshot'][0]['source_annotation_ids'], ['src1'])
            self.assertNotIn('lie_probability', rows[0])
            self.assertNotIn('video_rppg_context_heart_rate_bpm', rows[0])

    def test_missing_sync_suppresses_contact_but_retains_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            store.update(sid, markers=validate_markers({}))
            result = store.export_dataset(sid, 'no-sync')
            row = read_rows(result['csv_path'])[0]
            self.assertEqual(row['sync_status'], 'missing')
            self.assertEqual(row['signal_start_seconds'], '')
            self.assertEqual(row['contact_heart_rate_monitor_bpm'], '')
            self.assertEqual(row['signal_sample_count'], '0')
            self.assertEqual(row['speech_respondent_words'], '4')
            self.assertEqual(row['episode_contact_heart_rate_monitor_bpm'], '')
            self.assertEqual(row['question_contact_heart_rate_monitor_bpm'], '')
            self.assertEqual(row['waiting_contact_heart_rate_monitor_bpm'], '')
            self.assertIn('нет синхронизации', row['notes'])

    def test_stale_links_and_unsafe_names_do_not_create_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            for name in ('../session', '', '/tmp/out', '.csv', 'CON', 'a\\b', 'x:y'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    store.export_dataset(sid, name)
            answers = copy.deepcopy(store.read(sid)['answers'])
            answers[0]['turn_ids'] = ['unknown']
            store.update(sid, answers=answers)
            with self.assertRaisesRegex(ValueError, 'связи'):
                store.export_dataset(sid, 'broken')
            store.update(sid, answer_review_required=True)
            with self.assertRaisesRegex(ValueError, 'устареть'):
                store.export_dataset(sid, 'stale')
            self.assertEqual(list(store.directory(sid).glob('*.csv')), [])
            self.assertEqual(list(store.directory(sid).glob('*.metadata.json')), [])
            self.assertEqual(dataset_filename('  Фокин.CSV  '), 'Фокин.csv')

    def test_repeat_export_preserves_previous_file_and_marks_estimated_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            store.update(sid, sync_estimate=dict(review_required=True, uncertainty_seconds=2))
            first = store.export_dataset(sid, 'data')
            original = Path(first['csv_path']).read_bytes()
            second = store.export_dataset(sid, 'data')
            self.assertEqual(Path(second['csv_path']).name, 'data (2).csv')
            self.assertEqual(Path(first['csv_path']).read_bytes(), original)
            row = read_rows(second['csv_path'])[0]
            self.assertEqual(row['sync_status'], 'estimated_needs_review')
            self.assertEqual(row['sync_uncertainty_seconds'], '2')
            self.assertIn('предварительная', row['notes'])

    def test_phase_measurements_and_delta_use_distinct_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            directory = store.directory(sid)
            values = np.load(directory / 'signals.npy')
            values[:101, 1] = 60
            values[101:200, 1] = 66
            np.save(directory / 'signals.npy', values)
            result = store.export_dataset(sid, 'phases')
            row = read_rows(result['csv_path'])[0]
            self.assertEqual(float(row['question_contact_heart_rate_monitor_bpm']), 60)
            self.assertAlmostEqual(float(row['waiting_contact_heart_rate_monitor_bpm']), values[100:201, 1].mean())
            self.assertEqual(float(row['contact_heart_rate_monitor_bpm']), 72)
            self.assertEqual(float(row['delta_answer_minus_question_contact_heart_rate_monitor_bpm']), 12)
            self.assertAlmostEqual(float(row['episode_contact_heart_rate_monitor_bpm']), values[:901, 1].mean())
            self.assertEqual(float(row['speech_response_latency_seconds']), 1)
            self.assertAlmostEqual(float(row['speech_voice_rms']), .003271234567)

    def test_question_overlap_and_zero_waiting_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            meta = store.read(sid)
            meta['turns'][0]['end'] = 2.3
            store.update(sid, **{key: meta[key] for key in ('turns', 'answers')})
            row = read_rows(store.export_dataset(sid, 'overlap')['csv_path'])[0]
            self.assertEqual(row['episode_status'], 'question_overlap_clipped')
            self.assertEqual(float(row['question_phase_end_seconds']), 2)
            self.assertEqual(row['waiting_duration_seconds'], '')
            self.assertEqual(row['waiting_contact_heart_rate_monitor_bpm'], '')
            self.assertEqual(row['speech_response_latency_seconds'], '')
            self.assertAlmostEqual(float(row['question_overlap_seconds']), .3)
            meta['turns'][0]['end'] = 2
            store.update(sid, **{key: meta[key] for key in ('turns', 'answers')})
            row = read_rows(store.export_dataset(sid, 'zero')['csv_path'])[0]
            self.assertEqual(float(row['waiting_duration_seconds']), 0)
            self.assertEqual(row['waiting_contact_heart_rate_monitor_bpm'], '')

    def test_rppg_window_can_fit_episode_without_fitting_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            points = [dict(time=8.5, feature_version=2, rppg_bpm=75, rppg_quality=.3,
                           rppg_window_start=0, rppg_window_end=8.5)]
            (store.directory(sid) / 'video_features.json').write_text(json.dumps(points))
            row = read_rows(store.export_dataset(sid, 'rppg')['csv_path'])[0]
            self.assertEqual(float(row['episode_video_rppg_heart_rate_bpm']), 75)
            self.assertEqual(row['video_rppg_heart_rate_bpm'], '')
            self.assertEqual(row['question_video_rppg_heart_rate_bpm'], '')
            self.assertEqual(row['waiting_video_rppg_heart_rate_bpm'], '')

    def test_question_context_excludes_baseline_and_previous_response(self):
        meta = dict(turns=[dict(id='instruction', start=1, end=3, speaker='interviewer'),
                           dict(id='question', start=125, end=128, speaker='interviewer'),
                           dict(id='r', start=130, end=140, speaker='respondent')],
                    rest_intervals=[dict(start=3, end=125)])
        answer = dict(start=130, end=140, turn_ids=['r'], question_turn_ids=['instruction', 'question'])
        episode = answer_episode(meta, answer)
        self.assertEqual(episode['start'], 125)
        self.assertEqual(episode['question_turn_ids'], ['question'])
        self.assertEqual(episode['waiting'], dict(start=128, end=130))
        self.assertEqual(episode['status'], 'context_trimmed')
        meta['turns'].append(dict(id='previous', start=126, end=127, speaker='respondent'))
        episode = answer_episode(meta, answer)
        self.assertEqual(episode['start'], 127)
        answer['question_turn_ids'] = []
        episode = answer_episode(meta, answer)
        self.assertEqual(episode['status'], 'answer_only')
        self.assertEqual(episode['start'], 130)
        self.assertIsNone(episode['question'])

    def test_button_creates_named_dataset_in_background(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from neuronet_lab.desktop import MainWindow

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            store, sid = fixture(Path(tmp))
            with patch.object(MainWindow, '_error') as error:
                window = MainWindow(store)
                window.dataset_name.setText('Мой датасет')
                window.dataset_button.click()
                worker = window.active_task
                self.assertIsNotNone(worker)
                self.assertFalse(window.dataset_button.isEnabled())
                loop = QEventLoop()
                worker.finished.connect(loop.quit)
                QTimer.singleShot(10000, loop.quit)
                loop.exec()
                app.processEvents()
                self.assertIsNone(window.active_task)
                error.assert_not_called()
                self.assertTrue(window.dataset_button.isEnabled())
                self.assertIn('Сохранено эпизодов: 2', window.dataset_status.text())
                self.assertEqual(len(read_rows(store.directory(sid) / 'Мой датасет.csv')), 2)
                window.close()


if __name__ == '__main__':
    unittest.main()
