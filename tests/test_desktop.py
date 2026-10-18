import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from neuronet_lab.desktop import METRICS, MainWindow
from neuronet_lab.session import SessionStore, map_time, validate_markers


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_session_and_window_work_without_http_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            session = store.create("Проба")
            sid = session["id"]
            source = Path(tmp) / "input.csv"
            source.write_text(
                "Elapsed Time,ECG,PPG pulse,Heart Rate pulse\n"
                "00:00:00,0.2,81.5,72\n"
                "00:00:00.002,0.3,82.5,72\n"
                "00:00:00.004,0.4,83.5,72\n",
                encoding="utf-8",
            )
            store.import_bioradio(sid, source)
            self.assertEqual(store.signals(sid, 0, 1)["channels"]["PPG"], [81.5, 82.5, 83.5])
            markers = validate_markers({
                "video": {"baseline_start": 2, "dialog_start": 10},
                "signal": {"baseline_start": 3, "dialog_start": 11},
            })
            store.update(sid, markers=markers)
            self.assertEqual(map_time(5, markers["video"], markers["signal"]), 6)

            window = MainWindow(store)
            window.resize(1180, 700)
            window.show()
            self.app.processEvents()
            self.assertEqual(window.sid, sid)
            self.assertEqual(set(window.plot_widgets), {"ECG", "PPG", "HeartRate"})
            self.assertLess(window.play_button.y(), window.video_widget.y())
            self.assertGreater(window.lower_scroll.verticalScrollBar().maximum(), 0)
            self.assertEqual(window.turn_table.rowCount(), 0)
            window.add_turn()
            window.turn_table.item(0, 3).setText("Да, это правда")
            window.save_turns()
            self.assertEqual(store.read(sid)["turns"][0]["text"], "Да, это правда")
            window.start_edit.setText("0")
            window.end_edit.setText("1")
            window.annotation_name.setText("Первый вопрос")
            window.save_annotation()
            self.assertEqual(store.read(sid)["annotations"][0]["name"], "Первый вопрос")
            stats = store.stats(sid, 0, 1)
            self.assertIsNone(stats["lie_probability"])
            window._render_stats(stats)
            self.assertEqual(window.metrics.rowCount(), len(METRICS))
            window.close()


if __name__ == "__main__":
    unittest.main()
