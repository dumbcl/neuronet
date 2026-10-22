"""Native desktop viewer for frontal video and BioRadio sessions.

Run with: python -m neuronet_lab.desktop
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget, QAbstractItemView, QSizePolicy, QLayout,
)

from .session import MARKER_NAMES, SessionStore, validate_markers, validate_turns
from .video_features import load_video_points


STAGE_LABELS = ("Начало покоя", "Начало разговора", "Конец разговора", "Конец записи")
CHANNEL_COLORS = {
    "ECG": "#ef8d94", "PPG": "#5fd8c2", "RIP": "#829cf7", "GSR": "#e9c46a",
    "Flow": "#c9a4f6", "HeartRate": "#ffb67c", "SpO2": "#a9dc8f",
}
METRICS = (
    ("ЧСС по ЭКГ", "contact.ecg.heart_rate_bpm", "уд/мин"),
    ("ЧСС по контактной ФПГ", "contact.ppg.heart_rate_bpm", "уд/мин"),
    ("ЧСС по видео rPPG", "video.rppg_heart_rate_bpm", "уд/мин"),
    ("rPPG с учётом предыстории", "video.rppg_context_heart_rate_bpm", "уд/мин"),
    ("Разница ЭКГ / ФПГ", "comparison.ecg_ppg_delta_bpm", "уд/мин"),
    ("Разница ЭКГ / rPPG", "comparison.ecg_rppg_delta_bpm", "уд/мин"),
    ("Дыхание RIP", "contact.respiration_bpm", "/мин"),
    ("SpO₂", "contact.spo2_percent", "%"),
    ("GSR, среднее", "contact.gsr_mean", ""),
    ("ВСР SDNN по ЭКГ", "contact.ecg.sdnn_ms", "мс"),
    ("ВСР RMSSD по ЭКГ", "contact.ecg.rmssd_ms", "мс"),
    ("ВСР VLF по ЭКГ", "contact.ecg.vlf_power", "мс²"),
    ("ВСР LF по ЭКГ", "contact.ecg.lf_power", "мс²"),
    ("ВСР HF по ЭКГ", "contact.ecg.hf_power", "мс²"),
    ("LF/HF по ЭКГ", "contact.ecg.lf_hf", ""),
    ("ВСР SDNN по ФПГ", "contact.ppg.sdnn_ms", "мс"),
    ("ВСР RMSSD по ФПГ", "contact.ppg.rmssd_ms", "мс"),
    ("LF/HF по ФПГ", "contact.ppg.lf_hf", ""),
    ("Задержка ответа", "speech.response_latency_seconds", "с"),
    ("Длительность ответа", "speech.response_duration_seconds", "с"),
    ("Скорость речи", "speech.words_per_minute", "слов/мин"),
    ("Слоги в секунду", "speech.syllables_per_second", ""),
    ("Паузы внутри выбранного ответа", "speech.internal_pause_seconds", "с"),
    ("Основной тон", "speech.pitch_hz", "Гц"),
    ("Разброс основного тона (IQR)", "speech.pitch_iqr_hz", "Гц"),
    ("Энергия речи", "speech.voice_rms", ""),
    ("Наклон корпуса", "video.trunk_tilt_degrees", "°"),
    ("Скорость смещения рамки лица", "video.head_motion", "ширин лица/с"),
    ("Движение рук", "video.hand_motion", ""),
    ("Движение ног", "video.leg_motion", ""),
    ("Вероятность лжи", "lie_probability", ""),
)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02}:{minutes:02}:{sec:02}" if hours else f"{minutes:02}:{sec:02}"


def parse_time(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if not value:
        return None
    parts = value.split(":")
    if len(parts) > 3 or not all(part.replace(".", "", 1).isdigit() for part in parts):
        raise ValueError(f"Неверное время: {value}")
    result = 0.0
    for part in parts:
        result = result * 60 + float(part)
    return result


def nested(data: dict, path: str):
    value = data
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


class TaskThread(QThread):
    result = Signal(object)
    failure = Signal(str)
    progress = Signal(int)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action

    def run(self):
        try:
            self.result.emit(self.action(lambda value: self.progress.emit(int(value))))
        except Exception as exc:
            self.failure.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, store: SessionStore | None = None):
        super().__init__()
        self.store = store or SessionStore()
        self.sid: str | None = None
        self.meta: dict | None = None
        self.video_features: list[dict] = []
        self.video_feature_times: list[float] = []
        self.current_bio_text = ""
        self.active_task: TaskThread | None = None
        self.plot_widgets: dict[str, pg.PlotWidget] = {}
        self.last_plot_second = -1
        self._seeking = False
        self.setWindowTitle("Нейронет · анализ сессии")
        self.resize(1480, 940)
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self._connect()
        self._list_sessions()

    def _button(self, label: str, action, parent=None) -> QPushButton:
        button = QPushButton(label, parent)
        button.clicked.connect(action)
        return button

    def _build_ui(self):
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(230)
        self.video_widget.setMaximumHeight(460)
        self.player.setVideoOutput(self.video_widget)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 10)
        outer.setSpacing(10)
        toolbar = QHBoxLayout()
        title = QLabel("НЕЙРОНЕТ")
        title.setObjectName("title")
        toolbar.addWidget(title)
        self.sessions = QComboBox()
        self.sessions.setMinimumWidth(230)
        toolbar.addWidget(self.sessions, 1)
        self.new_button = self._button("Новая сессия", self.new_session)
        self.video_button = self._button("Добавить видео", self.add_video)
        self.csv_button = self._button("Добавить BioRadio CSV", self.add_csv)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.video_button)
        toolbar.addWidget(self.csv_button)
        outer.addLayout(toolbar)

        self.info = QLabel("Создайте сессию, затем добавьте фронтальное видео и CSV BioRadio.")
        self.info.setObjectName("info")
        outer.addWidget(self.info)

        main = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(main, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.play_button = self._button("▶ Пуск", self.toggle_play)
        self.play_button.setFixedWidth(94)
        controls.addWidget(self.play_button)
        controls.addWidget(self._button("−5 с", lambda: self.seek_relative(-5)))
        controls.addWidget(self._button("+5 с", lambda: self.seek_relative(5)))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        controls.addWidget(self.slider, 1)
        self.clock = QLabel("00:00 / 00:00")
        controls.addWidget(self.clock)
        left_layout.addLayout(controls)
        left_layout.addWidget(self.video_widget)

        self.lower_scroll = QScrollArea()
        self.lower_scroll.setWidgetResizable(True)
        self.lower_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.lower_scroll.setMinimumHeight(170)
        lower_host = QWidget()
        lower_layout = QVBoxLayout(lower_host)
        lower_layout.setContentsMargins(2, 8, 6, 4)
        lower_layout.setSpacing(14)
        self.current = QLabel("Текущий кадр и показания появятся после загрузки файлов.")
        self.current.setWordWrap(True)
        self.current.setObjectName("current")
        lower_layout.addWidget(self.current)

        marker_box = QGroupBox("Синхронизация этапов · время от начала каждого файла")
        marker_grid = QGridLayout(marker_box)
        marker_grid.addWidget(QLabel("Этап"), 0, 0)
        marker_grid.addWidget(QLabel("Видео"), 0, 1)
        marker_grid.addWidget(QLabel("BioRadio CSV"), 0, 3)
        self.marker_edits: dict[str, dict[str, QLineEdit]] = {"video": {}, "signal": {}}
        for row, (key, label) in enumerate(zip(MARKER_NAMES, STAGE_LABELS), 1):
            marker_grid.addWidget(QLabel(label), row, 0)
            for medium, column in (("video", 1), ("signal", 3)):
                edit = QLineEdit()
                edit.setPlaceholderText("мм:сс или секунды")
                self.marker_edits[medium][key] = edit
                marker_grid.addWidget(edit, row, column)
            cursor = self._button("← курсор", lambda checked=False, name=key: self.set_video_marker(name))
            marker_grid.addWidget(cursor, row, 2)
        marker_grid.addWidget(self._button("Сохранить метки", self.save_markers), 5, 3)
        lower_layout.addWidget(marker_box)

        chart_box = QGroupBox("Сигналы BioRadio · около текущего кадра")
        self.chart_layout = QVBoxLayout(chart_box)
        self.chart_layout.setContentsMargins(8, 12, 8, 8)
        self.chart_layout.setSpacing(12)
        self.chart_layout.addWidget(QLabel("После импорта CSV здесь появятся графики."))
        lower_layout.addWidget(chart_box)
        lower_layout.addStretch()
        self.lower_scroll.setWidget(lower_host)
        left_layout.addWidget(self.lower_scroll, 1)
        main.addWidget(left)

        right = QTabWidget()
        right.setMinimumWidth(390)
        main.addWidget(right)
        main.setSizes([960, 475])
        right.addTab(self._transcript_tab(), "Стенограмма")
        right.addTab(self._interval_tab(), "Интервал")
        right.addTab(self._annotation_tab(), "Разметка")
        self.right_tabs = right
        self.statusBar().showMessage("Готово")

        self.plot_timer = QTimer(self)
        self.plot_timer.setInterval(1000)
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start()

    def _transcript_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        intro = QLabel("Реплики привязаны ко времени видео. Двойной щелчок по строке переносит курсор.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.turn_table = QTableWidget(0, 4)
        self.turn_table.setHorizontalHeaderLabels(["Начало", "Конец", "Говорящий", "Текст"])
        self.turn_table.setColumnWidth(0, 70)
        self.turn_table.setColumnWidth(1, 70)
        self.turn_table.setColumnWidth(2, 110)
        self.turn_table.horizontalHeader().setStretchLastSection(True)
        self.turn_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.turn_table, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Добавить реплику", self.add_turn))
        buttons.addWidget(self._button("Удалить", self.delete_turn))
        buttons.addWidget(self._button("Сохранить", self.save_turns))
        layout.addLayout(buttons)
        self.audio_button = self._button("Автоматическая расшифровка", self.analyze_audio)
        layout.addWidget(self.audio_button)
        layout.addWidget(self._button("Пересчитать акустику по видео", self.refresh_acoustics))
        layout.addWidget(QLabel("Говорящий определяется приблизительно; проверьте текст и метки вручную."))
        return tab

    def _interval_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Выберите начало и конец на временной шкале видео."))
        grid = QGridLayout()
        self.start_edit = QLineEdit("0")
        self.end_edit = QLineEdit("10")
        grid.addWidget(QLabel("Начало"), 0, 0)
        grid.addWidget(self.start_edit, 0, 1)
        grid.addWidget(self._button("← курсор", lambda: self.start_edit.setText(str(round(self.position(), 2)))), 0, 2)
        grid.addWidget(QLabel("Конец"), 1, 0)
        grid.addWidget(self.end_edit, 1, 1)
        grid.addWidget(self._button("← курсор", lambda: self.end_edit.setText(str(round(self.position(), 2)))), 1, 2)
        layout.addLayout(grid)
        self.stats_button = self._button("Рассчитать интервал", self.analyze_interval)
        layout.addWidget(self.stats_button)
        self.interval_info = QLabel("Вероятность лжи появится только после обучения и проверки модели.")
        self.interval_info.setWordWrap(True)
        layout.addWidget(self.interval_info)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.metrics.horizontalHeader().setStretchLastSection(True)
        self.metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.metrics, 1)
        self.video_analysis_button = self._button("Анализ лица, позы и rPPG", self.analyze_video)
        layout.addWidget(self.video_analysis_button)
        return tab

    def _annotation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        annotation_hint = QLabel("Экспертные метки для выбранного интервала. Они не являются предсказанием модели.")
        annotation_hint.setWordWrap(True)
        layout.addWidget(annotation_hint)
        self.annotation_table = QTableWidget(0, 4)
        self.annotation_table.setHorizontalHeaderLabels(["Начало", "Конец", "Тип", "Название"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.annotation_table, 1)
        self.annotation_name = QLineEdit()
        self.annotation_name.setPlaceholderText("Вопрос или ответ")
        layout.addWidget(self.annotation_name)
        self.annotation_type = QComboBox()
        for label, value in (("Без метки", "unlabeled"), ("База", "base"), ("Правда", "truth"), ("Ложь", "lie")):
            self.annotation_type.addItem(label, value)
        layout.addWidget(self.annotation_type)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Сохранить интервал", self.save_annotation))
        buttons.addWidget(self._button("Удалить выбранный", self.delete_annotation))
        layout.addLayout(buttons)
        export_box = QGroupBox("Датасет для исследования")
        export_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        export_layout = QVBoxLayout(export_box)
        hint = QLabel("Одна строка — вопрос → ожидание → полный ответ. Метка правда/ложь относится к ответу. Показатели этапов сохраняются отдельно в папке сессии.")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        export_layout.addWidget(hint)
        self.answers_button = self._button("Сформировать полные ответы из интервалов", self.build_answers)
        self.answers_button.setToolTip("Связать реплики, паузы и вопросы с текущими экспертными метками. Неоднозначные ответы будут исключены из экспорта.")
        export_layout.addWidget(self.answers_button)
        self.answers_status = QLabel("Сначала сохраните интервалы и проверьте роли говорящих.")
        self.answers_status.setWordWrap(True)
        self.answers_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        export_layout.addWidget(self.answers_status)
        export_controls = QHBoxLayout()
        self.dataset_name = QLineEdit("dataset")
        self.dataset_name.setPlaceholderText("Имя датасета, например fokin")
        self.dataset_name.setAccessibleName("Имя датасета")
        self.dataset_name.setToolTip("Имя файла без пути. Расширение .csv добавляется автоматически.")
        export_controls.addWidget(self.dataset_name, 1)
        self.dataset_button = self._button("Сформировать датасет", self.export_dataset)
        export_controls.addWidget(self.dataset_button)
        export_layout.addLayout(export_controls)
        self.dataset_status = QLabel("CSV с показателями + JSON с описанием полей и качеством данных. Недоступные значения останутся пустыми.")
        self.dataset_status.setWordWrap(True)
        self.dataset_status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.dataset_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        export_layout.addWidget(self.dataset_status)
        layout.addWidget(export_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(tab)
        return scroll

    def _connect(self):
        self.sessions.currentIndexChanged.connect(self.on_session_changed)
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(self.on_duration)
        self.player.playbackStateChanged.connect(self.on_playback)
        self.player.errorOccurred.connect(lambda error, message: self._error(message) if message else None)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.slider.sliderReleased.connect(self.seek_from_slider)
        self.slider.valueChanged.connect(self.on_slider_value)
        self.turn_table.cellDoubleClicked.connect(self.seek_to_turn)
        self.annotation_table.cellDoubleClicked.connect(self.seek_to_annotation)

    def _error(self, message: str):
        QMessageBox.warning(self, "Нейронет", message)
        self.statusBar().showMessage(message, 8000)

    def _guard(self, action):
        try:
            return action()
        except Exception as exc:
            self._error(str(exc))
            return None

    def _list_sessions(self, selected: str | None = None):
        rows = self.store.list()
        self.sessions.blockSignals(True)
        self.sessions.clear()
        self.sessions.addItem("Выберите сессию", None)
        for row in rows:
            self.sessions.addItem(row["name"], row["id"])
        target = selected or (rows[0]["id"] if rows else None)
        if target:
            index = self.sessions.findData(target)
            self.sessions.setCurrentIndex(index if index >= 0 else 0)
        self.sessions.blockSignals(False)
        if target:
            self.load_session(target)

    def new_session(self):
        name, ok = QInputDialog.getText(self, "Новая сессия", "Название сессии:")
        if ok:
            meta = self.store.create(name.strip() or "Новая сессия")
            self._list_sessions(meta["id"])

    def on_session_changed(self):
        sid = self.sessions.currentData()
        if sid:
            self._guard(lambda: self.load_session(sid))

    def load_session(self, sid: str):
        self.player.stop()
        self.sid = sid
        self.meta = self.store.read(sid)
        self.video_features = []
        self.video_feature_times = []
        self.current_bio_text = ""
        source = self.store.directory(sid) / "video_features.json"
        if source.is_file():
            self.video_features = load_video_points(source, source.stat().st_mtime_ns)
            self.video_feature_times = [point.get("time", 0) for point in self.video_features]
        path = self.store.video_path(sid)
        self.player.setSource(QUrl.fromLocalFile(str(path)) if path else QUrl())
        self._render_meta()
        self._render_turns()
        self._render_annotations()
        self._render_answers_status()
        self.dataset_status.setText("CSV с показателями + JSON с описанием полей и качеством данных. Недоступные значения останутся пустыми.")
        self.dataset_status.setToolTip("")
        self.metrics.setRowCount(0)
        self.last_plot_second = -1
        self.refresh_plot()
        self.statusBar().showMessage(f"Открыта сессия «{self.meta['name']}»", 5000)

    def _render_meta(self):
        if not self.meta:
            return
        video = self.meta.get("video")
        signal = self.meta.get("signal")
        video_info = f"Видео: {video['filename']}" if video else "Видео не добавлено"
        signal_info = f"BioRadio: {signal['sample_count']:,} отсчётов, {signal['sample_rate_hz']} Гц" if signal else "CSV не добавлен"
        if signal and signal.get("malformed_rows"):
            signal_info += f" · строк с ошибками: {signal['malformed_rows']:,}"
        self.info.setText(f"{video_info}    ·    {signal_info}")
        for medium in ("video", "signal"):
            for name in MARKER_NAMES:
                value = self.meta.get("markers", {}).get(medium, {}).get(name)
                self.marker_edits[medium][name].setText("" if value is None else str(value))

    def _run_task(self, label: str, action, done):
        if self.active_task is not None:
            self._error("Дождитесь завершения текущей обработки")
            return
        thread = TaskThread(action, self)
        self.active_task = thread
        self.statusBar().showMessage(label)
        for button in (self.new_button, self.video_button, self.csv_button, self.audio_button, self.video_analysis_button, self.stats_button, self.dataset_button, self.dataset_name, self.answers_button):
            button.setEnabled(False)
        self.sessions.setEnabled(False)
        thread.progress.connect(lambda value: self.statusBar().showMessage(f"{label}: {value}%"))
        thread.result.connect(lambda result: self._guard(lambda: done(result)))
        thread.failure.connect(self._error)
        thread.finished.connect(self._task_finished)
        thread.start()

    def _task_finished(self):
        self.active_task = None
        for button in (self.new_button, self.video_button, self.csv_button, self.audio_button, self.video_analysis_button, self.stats_button, self.dataset_button, self.dataset_name, self.answers_button):
            button.setEnabled(True)
        self.sessions.setEnabled(True)
        self.statusBar().showMessage("Готово", 4000)

    def add_video(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        filename, _ = QFileDialog.getOpenFileName(self, "Фронтальное видео", "", "Видео (*.mp4 *.mov *.m4v *.webm)")
        if filename:
            sid = self.sid
            self.player.stop()
            self.player.setSource(QUrl())
            self._run_task("Копирование видео", lambda progress: self.store.import_video(sid, Path(filename)), lambda result: self.load_session(sid))

    def add_csv(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        filename, _ = QFileDialog.getOpenFileName(self, "BioRadio CSV", "", "CSV (*.csv);;Все файлы (*)")
        if filename:
            sid = self.sid
            self._run_task("Импорт BioRadio CSV", lambda progress: self.store.import_bioradio(sid, Path(filename)), lambda result: self.load_session(sid))

    def analyze_video(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        sid = self.sid
        self._run_task("Анализ видео", lambda progress: self.store.analyze_video(sid, progress), lambda result: self.load_session(sid))

    def analyze_audio(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        sid = self.sid
        self._run_task("Расшифровка речи", lambda progress: self.store.transcribe(sid, progress), lambda result: self.load_session(sid))

    def refresh_acoustics(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        if not self.save_turns():
            return
        sid = self.sid
        self._run_task("Пересчёт акустики", lambda progress: self.store.refresh_acoustics(sid, progress), lambda result: self.load_session(sid))

    def set_video_marker(self, name: str):
        self.marker_edits["video"][name].setText(str(round(self.position(), 2)))

    def save_markers(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        def action():
            markers = {medium: {name: parse_time(edit.text()) for name, edit in entries.items()} for medium, entries in self.marker_edits.items()}
            self.meta = self.store.update(self.sid, markers=validate_markers(markers))
            self._render_meta()
            self.refresh_plot()
            self.statusBar().showMessage("Метки синхронизации сохранены", 5000)
            return True
        return self._guard(action)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif self.sid and self.store.video_path(self.sid):
            self.player.play()

    def seek_relative(self, seconds: int):
        if self.sid and self.store.video_path(self.sid):
            self.player.setPosition(max(0, min(self.player.duration(), self.player.position() + seconds * 1000)))
            self.refresh_plot()
            self.update_current()

    def on_playback(self, state):
        self.play_button.setText("Ⅱ Пауза" if state == QMediaPlayer.PlaybackState.PlayingState else "▶ Пуск")
        if state != QMediaPlayer.PlaybackState.PlayingState:
            self.update_current()

    def position(self) -> float:
        return self.player.position() / 1000

    def on_duration(self, milliseconds: int):
        self.slider.setMaximum(max(0, milliseconds))
        self.clock.setText(f"{fmt_time(self.position())} / {fmt_time(milliseconds / 1000)}")
        if self.sid and self.meta and self.meta.get("video") and milliseconds > 0:
            previous = self.meta["video"].get("duration")
            if previous is None or abs(previous - milliseconds / 1000) > 0.1:
                video = dict(self.meta["video"], duration=milliseconds / 1000)
                self.meta = self.store.update(self.sid, video=video)

    def on_position(self, milliseconds: int):
        if not self._seeking:
            self.slider.blockSignals(True)
            self.slider.setValue(milliseconds)
            self.slider.blockSignals(False)
        self.clock.setText(f"{fmt_time(milliseconds / 1000)} / {fmt_time(self.player.duration() / 1000)}")
        self._highlight_turn(milliseconds / 1000)
        self._render_current_pose(milliseconds / 1000)

    def on_slider_value(self, value: int):
        if self._seeking:
            self.clock.setText(f"{fmt_time(value / 1000)} / {fmt_time(self.player.duration() / 1000)}")

    def seek_from_slider(self):
        self._seeking = False
        self.player.setPosition(self.slider.value())
        self.refresh_plot()
        self.update_current()

    def _render_current_pose(self, time: float):
        if not self.video_features:
            self.current.setText("Поза и rPPG: запустите анализ видео." + self.current_bio_text)
            return
        index = bisect_left(self.video_feature_times, time)
        candidates = self.video_features[max(0, index - 1):min(len(self.video_features), index + 1)]
        near = min(candidates, key=lambda point: abs(point.get("time", 0) - time))
        if abs(near.get("time", 0) - time) > 1:
            self.current.setText("Для текущего кадра нет рассчитанных признаков." + self.current_bio_text)
            return
        bits = []
        for label, key, unit in (("rPPG (скользящее окно)", "rppg_bpm", "уд/мин"), ("Корпус", "trunk_tilt", "°"), ("Смещение лица", "head_motion", "ширин/с"), ("Руки", "hand_motion", ""), ("Ноги", "leg_motion", "")):
            if near.get(key) is not None:
                bits.append(f"{label}: {near[key]:.5g} {unit}")
        if near.get("rppg_window_start") is not None:
            bits.append(f"Окно rPPG: {near['rppg_window_start']:.1f}–{near['rppg_window_end']:.1f} с")
        self.current.setText(("  ·  ".join(bits) or "Признаки текущего кадра недоступны") + self.current_bio_text)

    def update_current(self):
        if not self.sid or not self.meta or not self.meta.get("signal") or self.active_task:
            return
        time = self.position()
        def done(stats):
            contact = stats["contact"]
            pieces = []
            for label, value, unit in (("ЭКГ", nested(contact, "ecg.heart_rate_bpm"), "уд/мин"), ("ФПГ", nested(contact, "ppg.heart_rate_bpm"), "уд/мин"), ("SpO₂", contact.get("spo2_percent"), "%")):
                if value is not None:
                    pieces.append(f"{label}: {value} {unit}")
            if pieces:
                self.current_bio_text = "\nBioRadio: " + "  ·  ".join(pieces)
                self._render_current_pose(self.position())
                self.statusBar().showMessage("Текущие показания · " + "  ·  ".join(pieces), 8000)
        self._run_task("Текущие показания", lambda progress: self.store.stats(self.sid, max(0, time - 5), time + 5), done)

    def refresh_plot(self):
        if not self.sid or not self.meta or not self.meta.get("signal"):
            return
        second = int(self.position())
        if second == self.last_plot_second:
            return
        self.last_plot_second = second
        try:
            data = self.store.signals(self.sid, max(0, second - 10), second + 10)
            channels = data["channels"]
            if set(channels) != set(self.plot_widgets):
                while self.chart_layout.count():
                    item = self.chart_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.plot_widgets.clear()
                for name in channels:
                    plot = pg.PlotWidget(title=name)
                    plot.setMinimumHeight(200)
                    plot.showGrid(x=True, y=True, alpha=0.18)
                    plot.setMouseEnabled(x=True, y=False)
                    plot.getPlotItem().hideButtons()
                    self.chart_layout.addWidget(plot)
                    self.plot_widgets[name] = plot
            times = np.asarray(data["time"], dtype=float)
            for name, plot in self.plot_widgets.items():
                plot.clear()
                values = np.array([np.nan if value is None else value for value in channels[name]], dtype=float)
                if len(times):
                    plot.plot(times, values, pen=pg.mkPen(CHANNEL_COLORS.get(name, "#65d6c1"), width=1.3), connect="finite")
                    cursor = pg.InfiniteLine(pos=self.position(), angle=90, movable=False, pen=pg.mkPen("#ffffff88"))
                    plot.addItem(cursor)
                    plot.setXRange(max(0, second - 10), second + 10, padding=0)
        except Exception as exc:
            self.statusBar().showMessage(f"Графики: {exc}", 8000)

    def _render_turns(self):
        self.turn_table.setRowCount(0)
        for turn in (self.meta or {}).get("turns", []):
            self._append_turn(turn)

    def _append_turn(self, turn: dict):
        row = self.turn_table.rowCount()
        self.turn_table.insertRow(row)
        speaker = "Интервьюер" if turn.get("speaker") == "interviewer" else "Респондент"
        for column, value in enumerate((turn["start"], turn["end"], speaker, turn.get("text", ""))):
            item = QTableWidgetItem(str(value))
            if column == 2:
                item.setForeground(pg.mkColor("#829cf7" if speaker == "Интервьюер" else "#5fd8c2"))
            self.turn_table.setItem(row, column, item)
        self.turn_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, {
            **{key: turn[key] for key in ("id", "review_kind") if turn.get(key) is not None},
            "pause_seconds": turn.get("pause_seconds"), "acoustic": turn.get("acoustic") or {},
            "words": turn.get("words", []), "_source_text": turn.get("text", ""),
            "_source_bounds": [turn["start"], turn["end"]],
        })

    def add_turn(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        time = round(self.position(), 2)
        self._append_turn({"start": time, "end": time + 5, "speaker": "respondent", "text": ""})
        self.turn_table.scrollToBottom()

    def delete_turn(self):
        row = self.turn_table.currentRow()
        if row >= 0:
            self.turn_table.removeRow(row)

    def save_turns(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        def action():
            turns = []
            for row in range(self.turn_table.rowCount()):
                cells = [self.turn_table.item(row, col).text().strip() for col in range(4)]
                speaker = cells[2].lower()
                if speaker not in {"интервьюер", "респондент", "interviewer", "respondent"}:
                    raise ValueError(f"Строка {row + 1}: говорящий — Интервьюер или Респондент")
                extra = self.turn_table.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
                extra = dict(extra)
                bounds = [parse_time(cells[0]), parse_time(cells[1])]
                text_changed = extra.pop("_source_text", cells[3]) != cells[3]
                bounds_changed = extra.pop("_source_bounds", bounds) != bounds
                if text_changed or bounds_changed:
                    extra.update(words=[], pause_seconds=None)
                if bounds_changed:
                    extra["acoustic"] = {}
                turns.append({"start": bounds[0], "end": bounds[1], "speaker": "interviewer" if speaker in {"интервьюер", "interviewer"} else "respondent", "text": cells[3], **extra})
            self.meta = self.store.update(self.sid, turns=validate_turns(turns))
            self._render_turns()
            self._render_answers_status()
            self.statusBar().showMessage("Стенограмма сохранена", 5000)
            return True
        return self._guard(action)

    def _highlight_turn(self, time: float):
        for row in range(self.turn_table.rowCount()):
            try:
                start = parse_time(self.turn_table.item(row, 0).text())
                end = parse_time(self.turn_table.item(row, 1).text())
                active = start <= time <= end
            except (ValueError, TypeError):
                active = False
            for column in range(4):
                item = self.turn_table.item(row, column)
                if item:
                    item.setBackground(pg.mkColor("#294659" if active else "#18222e"))

    def seek_to_turn(self, row: int, column: int):
        self._guard(lambda: self.player.setPosition(int(parse_time(self.turn_table.item(row, 0).text()) * 1000)))

    def selection(self) -> tuple[float, float]:
        start = parse_time(self.start_edit.text())
        end = parse_time(self.end_edit.text())
        if start is None or end is None or end <= start:
            raise ValueError("Укажите интервал: конец должен быть позже начала")
        return start, end

    def analyze_interval(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        selection = self._guard(self.selection)
        if not selection:
            return
        start, end = selection
        sid = self.sid
        self._run_task("Расчёт интервала", lambda progress: self.store.stats(sid, start, end), self._render_stats)

    def _render_stats(self, result: dict):
        self.metrics.setRowCount(len(METRICS))
        for row, (label, path, unit) in enumerate(METRICS):
            value = nested(result, path)
            if isinstance(value, float):
                # Keep calculation precision; only format the displayed value.
                value = f"{value:.6g}"
            display = "—" if value is None else f"{value} {unit}".strip()
            self.metrics.setItem(row, 0, QTableWidgetItem(label))
            self.metrics.setItem(row, 1, QTableWidgetItem(display))
        self.metrics.resizeColumnToContents(0)
        self.interval_info.setText(
            f"Интервал {fmt_time(result['video_interval'][0])}–{fmt_time(result['video_interval'][1])} · "
            f"{result['sample_count']:,} отсчётов BioRadio. "
            + " ".join(result.get("notes", []))
            + " Вероятность лжи не вычисляется: модель пока не обучена."
        )
        self.right_tabs.setCurrentIndex(1)

    def _render_annotations(self):
        entries = (self.meta or {}).get("annotations", [])
        self.annotation_table.setRowCount(len(entries))
        labels = {"unlabeled": "Без метки", "base": "База", "truth": "Правда", "lie": "Ложь"}
        for row, item in enumerate(entries):
            for column, value in enumerate((item["start"], item["end"], labels.get(item["label"], item["label"]), item.get("name", ""))):
                self.annotation_table.setItem(row, column, QTableWidgetItem(str(value)))

    def save_annotation(self):
        if not self.sid:
            return self._error("Сначала создайте сессию")
        def action():
            start, end = self.selection()
            entries = list(self.meta.get("annotations", []))
            entries.append({"start": start, "end": end, "name": self.annotation_name.text().strip(), "label": self.annotation_type.currentData()})
            self.meta = self.store.update(self.sid, annotations=sorted(entries, key=lambda item: item["start"]))
            self._render_annotations()
            self._render_answers_status()
            self.statusBar().showMessage("Экспертная метка сохранена", 5000)
        self._guard(action)

    def delete_annotation(self):
        row = self.annotation_table.currentRow()
        if self.sid and row >= 0:
            entries = [item for index, item in enumerate(self.meta.get("annotations", [])) if index != row]
            self.meta = self.store.update(self.sid, annotations=entries)
            self._render_annotations()
            self._render_answers_status()

    def seek_to_annotation(self, row: int, column: int):
        item = self.meta["annotations"][row]
        self.start_edit.setText(str(item["start"]))
        self.end_edit.setText(str(item["end"]))
        self.player.setPosition(int(item["start"] * 1000))

    def _render_answers_status(self):
        meta = self.meta or {}
        self.answers_status.setToolTip("")
        if meta.get("answer_review_required"):
            self.answers_status.setText("Разметка или роли изменены. Сформируйте полные ответы заново перед выгрузкой.")
            return
        answers = meta.get("answers") or []
        if not answers:
            self.answers_status.setText("Полные ответы ещё не сформированы. Сохраните интервалы и проверьте роли говорящих.")
            return
        eligible = sum(bool(a.get("dataset_eligible")) for a in answers)
        review = sum(bool(a.get("review_required")) for a in answers)
        text = f"Полных ответов: {len(answers)}. Доступно для экспорта: {eligible}. Требуют проверки: {review}."
        report = meta.get("answers_build") or {}
        unlinked = report.get("unlinked_intervals", [])
        if unlinked:
            text += f" Интервалов без реплики респондента: {len(unlinked)}."
        problems = report.get("issues", []) + unlinked
        for issue in problems[:4]:
            text += f"\n{issue['start']:.2f}–{issue['end']:.2f} с: {issue['reason']}"
        if len(problems) > 4:
            text += f"\nЕщё замечаний: {len(problems)-4}. Полный список в отчёте."
        self.answers_status.setText(text)
        if report.get("report") and self.sid:
            self.answers_status.setToolTip(str(self.store.directory(self.sid) / report["report"]))

    def build_answers(self):
        if not self.sid:
            return self._error("Сначала выберите сессию")
        if self.active_task is not None:
            return self._error("Дождитесь завершения текущей обработки")
        if not self.save_turns() or not self.save_markers():
            return

        def action():
            # Honour edits visible in the interval table, retaining source IDs.
            labels = {"правда":"truth", "ложь":"lie", "база":"base", "без метки":"unlabeled",
                      "truth":"truth", "lie":"lie", "base":"base", "unlabeled":"unlabeled"}
            entries = []
            for row in range(self.annotation_table.rowCount()):
                cells = [self.annotation_table.item(row,col).text().strip() for col in range(4)]
                if cells[2].lower() not in labels:
                    raise ValueError(f"Интервал {row+1}: укажите База, Правда, Ложь или Без метки.")
                start, end = parse_time(cells[0]), parse_time(cells[1])
                if start is None or end is None or end <= start:
                    raise ValueError(f"Интервал {row+1}: конец должен быть позже начала.")
                item = dict(self.meta["annotations"][row], start=start, end=end, label=labels[cells[2].lower()], name=cells[3])
                entries.append(item)
            self.meta = self.store.update(self.sid, annotations=sorted(entries,key=lambda a:a["start"]))
            result = self.store.build_answers(self.sid)
            self.meta = result["session"]
            self._render_turns()
            self._render_annotations()
            self._render_answers_status()
            next_step = ("Теперь можно сформировать датасет." if result["report"]["eligible_count"]
                         else "Для экспорта уточните метки и замечания, затем сформируйте ответы повторно.")
            self.dataset_status.setText(f"Полные ответы сохранены. {next_step}\nРезервная копия и отчёт находятся в подпапке reviews сессии.")
            self.dataset_status.setToolTip(f"{result['backup_path']}\n{result['report_path']}")
            self.statusBar().showMessage("Полные ответы сформированы", 5000)
            return result
        return self._guard(action)

    def export_dataset(self):
        if not self.sid:
            return self._error("Сначала выберите сессию")
        if self.active_task is not None:
            return self._error("Дождитесь завершения текущей обработки")
        from .dataset import dataset_filename
        name = self._guard(lambda: dataset_filename(self.dataset_name.text()))
        if not name or not self.save_turns() or not self.save_markers():
            return
        sid = self.sid
        self.dataset_status.setText("Вычисление показателей полных ответов…")

        def done(result):
            text = (f"Сохранено эпизодов: {result['row_count']}. Исключено: {result['excluded_count']}.\n"
                    f"CSV: {Path(result['csv_path']).name}\nОписание: {Path(result['metadata_path']).name}")
            if result["sync_status"] == "missing":
                text += "\nBioRadio не включён: задайте синхронизацию с видео."
            elif result["sync_status"] == "estimated_needs_review":
                text += "\nСинхронизация BioRadio предварительная; замечание записано в датасет."
            self.dataset_status.setText(text)
            self.dataset_status.setToolTip(f"{result['csv_path']}\n{result['metadata_path']}")

        self._run_task("Выгрузка датасета", lambda progress: self.store.export_dataset(sid, name, progress), done)
        self.active_task.failure.connect(lambda message: self.dataset_status.setText(f"Выгрузка не выполнена: {message}"))

    def closeEvent(self, event):
        if self.active_task is not None:
            self._error("Дождитесь завершения обработки перед закрытием приложения")
            event.ignore()
            return
        self.player.stop()
        event.accept()


STYLE = """
QWidget { background: #101720; color: #e9f1fa; font-size: 13px; }
QMainWindow { background: #101720; }
QLabel#title { color: #65d6c1; font-size: 20px; font-weight: 800; letter-spacing: 2px; }
QLabel#info, QLabel#current { background: #1b2835; border-radius: 7px; padding: 9px; }
QGroupBox { border: 1px solid #344453; border-radius: 8px; margin-top: 12px; padding: 10px 7px 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fdccc; }
QPushButton { background: #263b4b; border: 1px solid #426074; border-radius: 6px; padding: 7px 10px; }
QPushButton:hover { background: #33566b; }
QPushButton:disabled { color: #71808f; background: #1d2934; }
QLineEdit, QComboBox, QTableWidget { background: #18222e; border: 1px solid #344453; border-radius: 5px; padding: 4px; selection-background-color: #325b66; }
QHeaderView::section { background: #253342; border: 0; border-right: 1px solid #344453; padding: 6px; }
QTabWidget::pane { border: 1px solid #344453; }
QTabBar::tab { background: #1b2835; padding: 9px 14px; }
QTabBar::tab:selected { background: #2b4755; color: #65d6c1; }
QSlider::groove:horizontal { height: 5px; background: #344453; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #65d6c1; border-radius: 7px; }
QStatusBar { background: #1b2835; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Нейронет")
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(background="#18222e", foreground="#adbdcc", antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
