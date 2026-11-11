"""Kdenlive Nvidia Renderer – PySide6 user interface."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from fractions import Fraction

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
                               QDoubleSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
                               QHeaderView, QStackedWidget)

from .engine.analyze import Report, analyze
from .engine.encode import CODECS, EncodeSettings
from .mlt.project import Project, load_project

APP_NAME = "Kdenlive Nvidia Renderer"

FPS_CHOICES = [("23.976", Fraction(24000, 1001)), ("24", Fraction(24)), ("25", Fraction(25)),
               ("29.97", Fraction(30000, 1001)), ("30", Fraction(30)), ("50", Fraction(50)),
               ("59.94", Fraction(60000, 1001)), ("60", Fraction(60))]
HEIGHTS = [2160, 1440, 1080, 720, 480]
STATUS_TEXT = {"gpu": "GPU", "audio": "Audio mix", "ignored": "Not needed", "unsupported": "Unsupported – skipped"}


def _fmt_time(sec: float | None) -> str:
    if sec is None:
        return "–"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------------------- worker

class RenderWorker(QObject):
    progress = Signal(object)
    log = Signal(str)
    finished = Signal(float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, settings: EncodeSettings, frame_range):
        super().__init__()
        self.project = project
        self.settings = settings
        self.frame_range = frame_range
        self.cancel_event = threading.Event()

    def run(self):
        try:
            from .engine.render import Cancelled, render
            elapsed = render(self.project, self.settings, self.frame_range,
                             progress=self.progress.emit, log=self.log.emit, cancel=self.cancel_event)
            self.finished.emit(elapsed)
        except Exception as e:  # noqa: BLE001 – report everything to the UI
            from .engine.render import Cancelled
            if isinstance(e, Cancelled):
                self.cancelled.emit()
            else:
                import traceback
                self.log.emit(traceback.format_exc())
                self.failed.emit(str(e))


class GpuProbe(QObject):
    done = Signal(str, list)

    def run(self):
        from .engine.media import FFMPEG
        gpu = "No NVIDIA GPU detected"
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                name, drv = [x.strip() for x in out.splitlines()[0].split(",")]
                gpu = f"{name} (driver {drv})"
        except Exception:
            pass
        encoders = []
        try:
            out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                 timeout=10).stdout
            encoders = [c for c in CODECS.values() if f" {c} " in out]
        except Exception:
            pass
        self.done.emit(gpu, encoders)


# ----------------------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self, initial: str | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon.fromTheme("kdenlive", QIcon.fromTheme("video-x-generic")))
        self.settings_store = QSettings("kdenlive-nvidia-renderer", "app")
        self.project: Project | None = None
        self.report: Report | None = None
        self.thread: QThread | None = None
        self.worker: RenderWorker | None = None
        self.available_encoders: list[str] = list(CODECS.values())
        self._build()
        self.resize(820, 860)
        self._probe_gpu()
        if initial:
            self.open_project(initial)

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # project picker
        g_proj = QGroupBox("Project")
        lp = QVBoxLayout(g_proj)
        row = QHBoxLayout()
        self.ed_project = QLineEdit()
        self.ed_project.setReadOnly(True)
        self.ed_project.setPlaceholderText("Choose a .kdenlive project…")
        btn_open = QPushButton(QIcon.fromTheme("document-open"), "Open…")
        btn_open.clicked.connect(self.choose_project)
        row.addWidget(self.ed_project, 1)
        row.addWidget(btn_open)
        lp.addLayout(row)
        self.lbl_info = QLabel("No project loaded.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lp.addWidget(self.lbl_info)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Effect / transition / clip type", "Uses", "Rendering"])
        self.tree.setRootIsDecorated(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.setMinimumHeight(150)
        lp.addWidget(self.tree)
        root.addWidget(g_proj)

        # render settings
        g_set = QGroupBox("Render settings")
        form = QFormLayout(g_set)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        res_row = QHBoxLayout()
        self.cb_res = QComboBox()
        self.cb_res.currentIndexChanged.connect(self._res_changed)
        self.sp_w = QSpinBox()
        self.sp_h = QSpinBox()
        for sp in (self.sp_w, self.sp_h):
            sp.setRange(16, 8192)
            sp.setSingleStep(2)
            sp.setEnabled(False)
        res_row.addWidget(self.cb_res, 1)
        res_row.addWidget(self.sp_w)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self.sp_h)
        form.addRow("Resolution:", res_row)

        self.cb_fps = QComboBox()
        form.addRow("Frame rate:", self.cb_fps)

        self.cb_range = QComboBox()
        form.addRow("Range:", self.cb_range)

        self.cb_codec = QComboBox()
        for label, codec in CODECS.items():
            self.cb_codec.addItem(label, codec)
        form.addRow("Video codec:", self.cb_codec)

        self.cb_preset = QComboBox()
        for i in range(1, 8):
            tag = {1: " (fastest)", 4: " (balanced)", 7: " (best quality)"}.get(i, "")
            self.cb_preset.addItem(f"P{i}{tag}", f"p{i}")
        self.cb_preset.setCurrentIndex(4)
        form.addRow("Encoder preset:", self.cb_preset)

        rc_row = QHBoxLayout()
        self.cb_rc = QComboBox()
        self.cb_rc.addItem("Constant quality", "cq")
        self.cb_rc.addItem("Target bitrate", "vbr")
        self.cb_rc.currentIndexChanged.connect(self._rc_changed)
        self.stack_rc = QStackedWidget()
        self.sp_cq = QSpinBox()
        self.sp_cq.setRange(1, 51)
        self.sp_cq.setValue(19)
        self.sp_cq.setPrefix("CQ ")
        self.sp_cq.setToolTip("Lower = better quality / bigger file. 16–23 is typical.")
        self.sp_br = QDoubleSpinBox()
        self.sp_br.setRange(0.5, 400)
        self.sp_br.setValue(20)
        self.sp_br.setSuffix(" Mbit/s")
        self.stack_rc.addWidget(self.sp_cq)
        self.stack_rc.addWidget(self.sp_br)
        self.stack_rc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        rc_row.addWidget(self.cb_rc, 1)
        rc_row.addWidget(self.stack_rc, 1)
        form.addRow("Quality:", rc_row)

        au_row = QHBoxLayout()
        self.chk_audio = QCheckBox("Include audio")
        self.chk_audio.setChecked(True)
        self.cb_abr = QComboBox()
        for k in (128, 192, 256, 320):
            self.cb_abr.addItem(f"AAC {k} kbit/s", k)
        self.cb_abr.setCurrentIndex(3)
        self.chk_audio.toggled.connect(self.cb_abr.setEnabled)
        au_row.addWidget(self.chk_audio)
        au_row.addWidget(self.cb_abr, 1)
        form.addRow("Audio:", au_row)

        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Output file…")
        btn_out = QPushButton(QIcon.fromTheme("document-save-as"), "Browse…")
        btn_out.clicked.connect(self.choose_output)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_out)
        form.addRow("Output:", out_row)
        root.addWidget(g_set)

        # render controls
        g_run = QGroupBox("Render")
        lr = QVBoxLayout(g_run)
        btn_row = QHBoxLayout()
        self.btn_render = QPushButton(QIcon.fromTheme("media-record"), "Render")
        f = QFont(self.btn_render.font())
        f.setBold(True)
        self.btn_render.setFont(f)
        self.btn_render.setMinimumHeight(36)
        self.btn_render.setEnabled(False)
        self.btn_render.clicked.connect(self.start_render)
        self.btn_cancel = QPushButton(QIcon.fromTheme("process-stop"), "Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_render)
        self.btn_folder = QPushButton(QIcon.fromTheme("folder-open"), "Open folder")
        self.btn_folder.setEnabled(False)
        self.btn_folder.clicked.connect(self.open_folder)
        btn_row.addWidget(self.btn_render, 2)
        btn_row.addWidget(self.btn_cancel, 1)
        btn_row.addWidget(self.btn_folder, 1)
        lr.addLayout(btn_row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        lr.addWidget(self.bar)
        self.lbl_status = QLabel("Idle")
        lr.addWidget(self.lbl_status)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMinimumHeight(90)
        lr.addWidget(self.log_view)
        root.addWidget(g_run, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Checking GPU…")
        self._restore_settings()

    # ------------------------------------------------------------------ persistence
    def _restore_settings(self):
        s = self.settings_store
        codec = s.value("codec", "h264_nvenc")
        i = self.cb_codec.findData(codec)
        if i >= 0:
            self.cb_codec.setCurrentIndex(i)
        preset = s.value("preset", "p5")
        i = self.cb_preset.findData(preset)
        if i >= 0:
            self.cb_preset.setCurrentIndex(i)
        self.cb_rc.setCurrentIndex(1 if s.value("rc", "cq") == "vbr" else 0)
        self.sp_cq.setValue(int(s.value("cq", 19)))
        self.sp_br.setValue(float(s.value("bitrate", 20)))
        self.chk_audio.setChecked(s.value("audio", "true") in (True, "true"))
        i = self.cb_abr.findData(int(s.value("abr", 320)))
        if i >= 0:
            self.cb_abr.setCurrentIndex(i)

    def _save_settings(self):
        s = self.settings_store
        s.setValue("codec", self.cb_codec.currentData())
        s.setValue("preset", self.cb_preset.currentData())
        s.setValue("rc", self.cb_rc.currentData())
        s.setValue("cq", self.sp_cq.value())
        s.setValue("bitrate", self.sp_br.value())
        s.setValue("audio", self.chk_audio.isChecked())
        s.setValue("abr", self.cb_abr.currentData())

    # ------------------------------------------------------------------ gpu probe
    def _probe_gpu(self):
        self._probe_thread = QThread(self)
        self._probe = GpuProbe()
        self._probe.moveToThread(self._probe_thread)
        self._probe_thread.started.connect(self._probe.run)
        self._probe.done.connect(self._gpu_ready)
        self._probe.done.connect(self._probe_thread.quit)
        self._probe_thread.start()

    def _gpu_ready(self, gpu: str, encoders: list):
        self.statusBar().showMessage(f"GPU: {gpu}   ·   NVENC: {', '.join(encoders) or 'not available'}")
        self.available_encoders = encoders
        model = self.cb_codec.model()
        for i in range(self.cb_codec.count()):
            item = model.item(i)
            item.setEnabled(self.cb_codec.itemData(i) in encoders)
        if encoders and self.cb_codec.currentData() not in encoders:
            self.cb_codec.setCurrentIndex(self.cb_codec.findData(encoders[0]))

    # ------------------------------------------------------------------ project
    def choose_project(self):
        start = self.settings_store.value("last_project_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open Kdenlive project", start,
                                              "Kdenlive projects (*.kdenlive);;All files (*)")
        if path:
            self.open_project(path)

    def open_project(self, path: str):
        try:
            project = load_project(path)
            report = analyze(project)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Could not open project:\n{e}")
            return
        self.project, self.report = project, report
        path = project.path
        self.settings_store.setValue("last_project_dir", os.path.dirname(path))
        self.ed_project.setText(path)
        p = project.profile
        dur = report.duration_frames / p.fps if p.fps else 0
        info = (f"<b>{os.path.basename(path)}</b><br>"
                f"Profile: {report.profile}<br>"
                f"Duration: {_fmt_time(dur)} ({report.duration_frames} frames) · "
                f"{report.tracks_video} video / {report.tracks_audio} audio tracks · {report.clips} clips")
        if report.missing_files:
            info += "<br><span style='color:#d9534f'><b>Missing media:</b> " + \
                    "<br>".join(report.missing_files) + "</span>"
        if report.unsupported:
            info += ("<br><span style='color:#e0a000'>Some items are not supported yet and will be "
                     "left out of the render (see list).</span>")
        self.lbl_info.setText(info)
        self._fill_tree(report)
        self._fill_resolutions(project)
        self._fill_fps(project)
        self._fill_ranges(project, report)
        base = os.path.splitext(path)[0]
        out_dir = self.settings_store.value("last_output_dir", "")
        if not out_dir or not os.path.isdir(out_dir):
            out_dir = os.path.dirname(path)
        self.ed_out.setText(os.path.join(out_dir, os.path.basename(base) + ".mp4"))
        self.btn_render.setEnabled(True)
        self.btn_folder.setEnabled(False)

    def _fill_tree(self, rep: Report):
        self.tree.clear()
        colors = {"gpu": Qt.darkGreen, "audio": Qt.darkCyan, "ignored": Qt.gray, "unsupported": Qt.darkRed}
        for title, items in (("Effects", rep.effects), ("Transitions", rep.transitions),
                             ("Clip types", rep.producers)):
            if not items:
                continue
            top = QTreeWidgetItem([f"{title} ({len(items)})", "", ""])
            self.tree.addTopLevelItem(top)
            for it in items:
                name = it.name if it.name == it.service else f"{it.name}  [{it.service}]"
                child = QTreeWidgetItem([name, str(it.count), STATUS_TEXT.get(it.status, it.status)])
                child.setForeground(2, colors.get(it.status, Qt.black))
                top.addChild(child)
            top.setExpanded(True)

    def _fill_resolutions(self, project: Project):
        p = project.profile
        self.cb_res.blockSignals(True)
        self.cb_res.clear()
        self.cb_res.addItem(f"Project ({p.width}×{p.height})", (p.width, p.height))
        for h in HEIGHTS:
            w = int(round(h * p.dar / 2)) * 2
            if (w, h) != (p.width, p.height):
                self.cb_res.addItem(f"{h}p ({w}×{h})", (w, h))
        self.cb_res.addItem("Custom…", None)
        self.cb_res.blockSignals(False)
        self.sp_w.setValue(p.width)
        self.sp_h.setValue(p.height)
        self._res_changed()

    def _fill_fps(self, project: Project):
        p = project.profile
        self.cb_fps.clear()
        self.cb_fps.addItem(f"Project ({p.fps:.3f} fps)", None)
        for label, fr in FPS_CHOICES:
            if fr != p.fps_fraction:
                self.cb_fps.addItem(f"{label} fps", fr)

    def _fill_ranges(self, project: Project, rep: Report):
        self.cb_range.clear()
        self.cb_range.addItem(f"Whole timeline ({_fmt_time(rep.duration_frames / project.fps)})", None)
        if rep.zone:
            zi, zo = rep.zone
            self.cb_range.addItem(f"Timeline zone ({_fmt_time(zi / project.fps)} – {_fmt_time(zo / project.fps)})",
                                  (zi, zo - 1))

    def _res_changed(self):
        data = self.cb_res.currentData()
        custom = data is None and self.cb_res.count() > 0
        self.sp_w.setEnabled(custom)
        self.sp_h.setEnabled(custom)
        if data:
            self.sp_w.setValue(data[0])
            self.sp_h.setValue(data[1])

    def _rc_changed(self):
        self.stack_rc.setCurrentIndex(self.cb_rc.currentIndex())

    def choose_output(self):
        current = self.ed_out.text() or os.path.expanduser("~/output.mp4")
        path, flt = QFileDialog.getSaveFileName(
            self, "Render to", current,
            "MP4 video (*.mp4);;Matroska video (*.mkv);;QuickTime video (*.mov)")
        if path:
            if not os.path.splitext(path)[1]:
                path += {"Matroska": ".mkv", "QuickTime": ".mov"}.get(flt.split(" ")[0], ".mp4")
            self.ed_out.setText(path)

    # ------------------------------------------------------------------ rendering
    def _settings(self) -> EncodeSettings | None:
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, APP_NAME, "Please choose an output file.")
            return None
        out = os.path.abspath(os.path.expanduser(out))
        if os.path.splitext(out)[1].lower() not in (".mp4", ".mkv", ".mov", ".m4v"):
            QMessageBox.warning(self, APP_NAME, "Output must be an .mp4, .mkv or .mov file.")
            return None
        if not os.path.isdir(os.path.dirname(out)):
            QMessageBox.warning(self, APP_NAME, f"Folder does not exist:\n{os.path.dirname(out)}")
            return None
        if self.project and os.path.abspath(out) == os.path.abspath(self.project.path):
            QMessageBox.warning(self, APP_NAME, "The output would overwrite the project file.")
            return None
        if os.path.exists(out):
            r = QMessageBox.question(self, APP_NAME, f"{os.path.basename(out)} already exists. Overwrite it?")
            if r != QMessageBox.Yes:
                return None
        w, h = self.sp_w.value(), self.sp_h.value()
        if w % 2 or h % 2:
            w, h = w - w % 2, h - h % 2
        fps = self.cb_fps.currentData() or self.project.profile.fps_fraction
        return EncodeSettings(output=out, width=w, height=h, fps=fps, codec=self.cb_codec.currentData(),
                              preset=self.cb_preset.currentData(), rate_control=self.cb_rc.currentData(),
                              quality=self.sp_cq.value(), bitrate_mbps=self.sp_br.value(),
                              audio=self.chk_audio.isChecked(), audio_bitrate_k=self.cb_abr.currentData())

    def start_render(self):
        if not self.project:
            return
        if self.report and self.report.missing_files:
            r = QMessageBox.question(self, APP_NAME, "Some media files are missing and will render as "
                                     "empty/black. Render anyway?")
            if r != QMessageBox.Yes:
                return
        settings = self._settings()
        if settings is None:
            return
        self._save_settings()
        self.settings_store.setValue("last_output_dir", os.path.dirname(settings.output))
        self.log_view.clear()
        self.bar.setValue(0)
        self._set_running(True)
        self._t_start = time.monotonic()
        self.thread = QThread(self)
        self.worker = RenderWorker(self.project, settings, self.cb_range.currentData())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        for sig in (self.worker.finished, self.worker.failed, self.worker.cancelled):
            sig.connect(self.thread.quit)
        self._last_output = settings.output
        self.lbl_status.setText("Starting…")
        self.thread.start()

    def cancel_render(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.lbl_status.setText("Cancelling…")
            self.btn_cancel.setEnabled(False)

    def _set_running(self, running: bool):
        self.btn_render.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_folder.setEnabled(False if running else self.btn_folder.isEnabled())
        for w in (self.cb_res, self.cb_fps, self.cb_range, self.cb_codec, self.cb_preset, self.cb_rc,
                  self.stack_rc, self.chk_audio, self.cb_abr, self.ed_out):
            w.setEnabled(not running)
        if not running:
            self._res_changed()
            self.cb_abr.setEnabled(self.chk_audio.isChecked())

    def _on_progress(self, p):
        if p.stage == "audio":
            frac = p.done / max(1, p.total)
            self.bar.setValue(int(frac * 50))
            self.lbl_status.setText(f"Mixing audio… {frac * 100:.0f}%")
        elif p.stage == "video":
            frac = p.done / max(1, p.total)
            self.bar.setValue(50 + int(frac * 950))
            self.lbl_status.setText(f"Rendering frame {p.done} / {p.total}  ·  {p.fps:.0f} fps  ·  "
                                    f"elapsed {_fmt_time(p.elapsed)}  ·  remaining {_fmt_time(p.eta)}")
        elif p.stage == "finishing":
            self.bar.setValue(995)
            self.lbl_status.setText("Finalising file…")

    def _on_log(self, msg: str):
        self.log_view.appendPlainText(msg)

    def _on_finished(self, elapsed: float):
        self._set_running(False)
        self.bar.setValue(1000)
        self.btn_folder.setEnabled(True)
        size = os.path.getsize(self._last_output) / 1e6 if os.path.exists(self._last_output) else 0
        self.lbl_status.setText(f"Done in {_fmt_time(elapsed)}  ·  {os.path.basename(self._last_output)} "
                                f"({size:.1f} MB)")
        self._on_log(f"Finished in {elapsed:.1f}s -> {self._last_output}")

    def _on_failed(self, msg: str):
        self._set_running(False)
        self.lbl_status.setText("Render failed")
        QMessageBox.critical(self, APP_NAME, f"Render failed:\n\n{msg[-2000:]}")

    def _on_cancelled(self):
        self._set_running(False)
        self.bar.setValue(0)
        self.lbl_status.setText("Cancelled")

    def open_folder(self):
        if getattr(self, "_last_output", None):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self._last_output)))

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            r = QMessageBox.question(self, APP_NAME, "A render is in progress. Cancel it and quit?")
            if r != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel_event.set()
            self.thread.quit()
            self.thread.wait(10000)
        event.accept()


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if not os.environ.get("QT_QPA_PLATFORMTHEME"):
        os.environ["QT_QPA_PLATFORMTHEME"] = "gtk3"      # native GTK file dialogs on Cinnamon/GNOME
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName("kdenlive-nvidia-renderer")
    initial = next((a for a in argv[1:] if a.endswith(".kdenlive")), None)
    win = MainWindow(initial)
    win.show()
    return app.exec()
