from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core import engine, sources
from ..core.gpu import cuda_info
from ..core.models import CAPTION_STYLES, JobConfig
from ..core.paths import config_dir, shorts_dir
from ..core.store import ConnectionStore, PresetStore, resolve_llm_for_preset
from ..core.subtitles import MODES as SUB_MODES
from ..core.transcribe import WHISPER_CHOICES
from . import theme, workers
from .dialogs import ConnectionDialog, ask_text
from .icons import app_icon, icon
from .widgets import (Chips, ClipCard, DropZone, Field, Steps, Toast, button, dot_text, hline, label, open_path)

STAGE_NAMES = [n for n, _ in engine.STAGES]
STAGE_SHORT = ["Get video", "Analyze", "Transcribe", "Find moments", "AI review", "Captions", "Make clips"]
STAGE_LONG = {"Import": "Getting the video", "Analyze": "Analyzing audio and scenes",
              "Transcribe": "Transcribing speech", "Find moments": "Finding the best moments",
              "AI review": "AI is choosing and titling clips", "Captions": "Making accurate captions",
              "Render": "Making your clips"}
LENGTHS = [((15, 30), "15–30s"), ((20, 60), "20–60s"), ((30, 90), "30–90s"), ((60, 180), "1–3 min"),
           ("custom", "Custom", "Choose your own minimum and maximum")]
COUNTS = list(range(1, 11))
ASPECT_OPTS = [("9:16", "9:16", "Shorts, Reels, TikTok"), ("1:1", "1:1"), ("4:5", "4:5", "Instagram feed"),
               ("16:9", "16:9"), ("Original", "Original"), ("Custom", "Custom")]
FRAMING = [("Auto", "Auto", "Fit for wide gameplay, fill otherwise"),
           ("Fit", "Fit", "Whole frame over a blurred background"),
           ("Fill", "Fill", "Crop to fill, centred"),
           ("Smart", "Follow action", "Crop that follows motion in the clip")]


def _clock(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _clock_left(sec: float) -> str:
    if sec < 60:
        return "1 min"
    m = int(round(sec / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def scroll(widget: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setWidget(widget)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return s


def _swatch(hx: str):
    from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
    pm = QPixmap(20, 20); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(hx)); p.drawEllipse(2, 2, 16, 16); p.end()
    return QIcon(pm)


def page_container() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    w.setObjectName("page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(48, 38, 48, 48)
    lay.setSpacing(0)
    return w, lay


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.resize(1180, 820)
        self.setMinimumSize(920, 640)
        self.qs = QSettings("MyClipAI", "MyClipAI")
        self.conns = ConnectionStore(config_dir() / "connections.json")
        self.presets = PresetStore(config_dir() / "presets.json")
        self.source = ""
        self.worker: workers.JobWorker | None = None
        self.live: workers.LiveWorker | None = None
        self.session_keys: dict[str, str] = {}
        self._loading = False

        root = QWidget(); root.setObjectName("root")
        h = QHBoxLayout(root); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        h.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.stack.addWidget(self._create_page())
        self.stack.addWidget(self._live_page())
        self.stack.addWidget(self._library_page())
        self.toast = Toast(self)
        from .studio import StudioPage
        self.studio = StudioPage(lambda t, ms=2600, error=False: self.toast.show_text(t, ms, error), self.conns.keys)
        self.stack.addWidget(scroll(self.studio))
        self.stack.addWidget(self._settings_page())
        ClipCard.upload_handler = self._upload

        self._restore()
        self._refresh_connections()
        self._refresh_presets()
        self._go(0)

        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.browse)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.generate)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=lambda: (self._go(0), self.url.setFocus()))
        QShortcut(QKeySequence("Ctrl+F"), self, activated=lambda: (self._go(2), self.lib_search.setFocus()))
        QShortcut(QKeySequence("Esc"), self, activated=self.cancel)
        for i in range(5):
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda i=i: self._go(i))

        QTimer.singleShot(400, self._check_env)
        QTimer.singleShot(700, lambda: self.detect_local(False))
        self._detect_timer = QTimer(self, interval=60000, timeout=lambda: self.detect_local(False, quiet=True))
        self._detect_timer.start()

    # ================= sidebar =================
    def _sidebar(self) -> QWidget:
        w = QWidget(); w.setObjectName("sidebar"); w.setFixedWidth(208)
        v = QVBoxLayout(w); v.setContentsMargins(12, 20, 12, 16); v.setSpacing(4)
        brand = QHBoxLayout(); brand.setSpacing(9); brand.setContentsMargins(6, 0, 0, 0)
        self.brand_icon = QLabel(); self.brand_icon.setPixmap(app_icon().pixmap(22, 22)); brand.addWidget(self.brand_icon)
        brand.addWidget(label(APP_NAME, "brand")); brand.addStretch(1)
        v.addLayout(brand); v.addSpacing(22)
        self.nav = []
        self.nav_icons = []
        for i, (name, ic_name) in enumerate([("Create", "create"), ("Live", "live"), ("Library", "library"), ("Studio", "studio"), ("Settings", "settings")]):
            b = QPushButton(name); b.setObjectName("nav"); b.setCheckable(True)
            b.setIcon(icon(ic_name, on_color=theme.ACCENT)); b.setIconSize(QSize(16, 16))
            self.nav_icons.append(ic_name)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, i=i: self._go(i))
            v.addWidget(b); self.nav.append(b)
        v.addStretch(1)
        self.device_lbl = label("", "faint", wrap=True)
        v.addWidget(self.device_lbl)
        v.addSpacing(4)
        v.addWidget(label(f"v{__version__}", "faint"))
        return w

    def _go(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        for j, b in enumerate(self.nav):
            b.setChecked(i == j)
        if i == 2:
            self._load_library()
        if i == 3 and self.studio.store.active:
            self.studio.refresh()

    # ================= create =================
    def _create_page(self) -> QWidget:
        page, v = page_container()
        head = QHBoxLayout()
        head.addWidget(label("New clips", "h1")); head.addStretch(1)
        v.addLayout(head)
        v.addSpacing(4)
        v.addWidget(label("Add a long video. MyClip finds the best moments and cuts them into shorts.", "muted"))
        v.addSpacing(22)

        self.drop = DropZone()
        self.drop.clicked.connect(self.browse)
        v.addWidget(self.drop)
        v.addSpacing(10)
        row = QHBoxLayout(); row.setSpacing(8)
        self.url = QLineEdit(); self.url.setObjectName("url")
        self.url.setPlaceholderText("Or paste a link  (YouTube, Twitch, Kick, X, ...)")
        self.url.returnPressed.connect(lambda: self._set_source(self.url.text()))
        self.url.editingFinished.connect(lambda: self.url.text().strip() and self._set_source(self.url.text()))
        row.addWidget(self.url, 1)
        self.gen_btn = button("Generate clips", "primary", on=self.generate)
        self.gen_btn.setMinimumHeight(38)
        self.gen_btn.setToolTip("Ctrl+Enter")
        row.addWidget(self.gen_btn)
        v.addLayout(row)
        self.summary_bar = QWidget()
        sb_l = QHBoxLayout(self.summary_bar); sb_l.setContentsMargins(0, 14, 0, 0); sb_l.setSpacing(10)
        self.summary_lbl = label("", "faint"); self.summary_lbl.setWordWrap(True)
        sb_l.addWidget(self.summary_lbl, 1)
        sb_l.addWidget(button("Edit settings", "ghost", "down", on=lambda: self._collapse_options(False, scroll=True)),
                       0, Qt.AlignmentFlag.AlignTop)
        v.addWidget(self.summary_bar)
        self.summary_bar.hide()
        v.addSpacing(28)

        # all options live in one box that folds away while a job runs
        self.options_box = QWidget()
        ov = QVBoxLayout(self.options_box); ov.setContentsMargins(0, 0, 0, 0); ov.setSpacing(0)
        # presets
        pr = QHBoxLayout(); pr.setSpacing(10)
        pr.addWidget(label("PRESET", "section")); pr.addStretch(1)
        self.save_preset_btn = button("Save current", "ghost", on=self.save_preset)
        pr.addWidget(self.save_preset_btn)
        ov.addLayout(pr); ov.addSpacing(8)
        self.preset_chips = Chips()
        self.preset_chips.changed.connect(self.apply_preset)
        self.preset_chips.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preset_chips.customContextMenuRequested.connect(self._preset_menu)
        ov.addWidget(self.preset_chips)
        ov.addSpacing(24)

        # main options grid
        grid = QGridLayout(); grid.setHorizontalSpacing(36); grid.setVerticalSpacing(22)
        self.count = Chips([(c, str(c)) for c in COUNTS])
        self.length = Chips(LENGTHS)
        self.aspect = Chips(ASPECT_OPTS)
        self.captions = Chips(list(CAPTION_STYLES) + ["Off"])
        self.cap_lang = Chips(SUB_MODES)
        self.captions.changed.connect(lambda v: self.cap_lang.setEnabled(v != "Off"))
        grid.addWidget(Field("Number of clips", self.count), 0, 0, 1, 2)
        grid.addWidget(Field("Length", self.length), 1, 0)
        grid.addWidget(Field("Format", self.aspect), 1, 1)
        grid.addWidget(Field("Captions", self.captions), 2, 0)
        grid.addWidget(Field("Subtitle language", self.cap_lang,
                             "Auto: Hindi becomes Hinglish (“ye kya ho gaya”), English stays English, and a word "
                             "from another language gets its meaning in brackets, like “gracias (thanks)”. "
                             "Brackets need an AI model."), 2, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        ov.addLayout(grid)
        self.len_row = QWidget()
        lr = QHBoxLayout(self.len_row); lr.setContentsMargins(0, 12, 0, 0); lr.setSpacing(8)
        self.len_min = QSpinBox(); self.len_min.setRange(5, 595); self.len_min.setSuffix(" s"); self.len_min.setValue(20)
        self.len_max = QSpinBox(); self.len_max.setRange(10, 600); self.len_max.setSuffix(" s"); self.len_max.setValue(60)
        for sb in (self.len_min, self.len_max):
            sb.setMinimumWidth(84); sb.setAccelerated(True)
            sb.setToolTip("Type a number or use the mouse wheel")
        lr.addWidget(label("Clip length from", "muted")); lr.addWidget(self.len_min)
        lr.addWidget(label("to", "muted")); lr.addWidget(self.len_max)
        self.len_hint = label("", "faint"); lr.addSpacing(8); lr.addWidget(self.len_hint)
        lr.addStretch(1)
        ov.addWidget(self.len_row)
        self.len_min.valueChanged.connect(self._len_min_changed)
        self.len_max.valueChanged.connect(self._len_max_changed)
        self.length.changed.connect(lambda v: (self.len_row.setVisible(v == "custom"), self._len_hint()))
        self.custom_row = QWidget()
        cr = QHBoxLayout(self.custom_row); cr.setContentsMargins(0, 10, 0, 0); cr.setSpacing(8)
        self.cw = QSpinBox(); self.cw.setRange(240, 4096); self.cw.setSingleStep(2); self.cw.setValue(1080)
        self.ch = QSpinBox(); self.ch.setRange(240, 4096); self.ch.setSingleStep(2); self.ch.setValue(1920)
        cr.addWidget(label("Size", "muted")); cr.addWidget(self.cw); cr.addWidget(label("×", "faint")); cr.addWidget(self.ch)
        cr.addStretch(1)
        ov.addWidget(self.custom_row)
        self.aspect.changed.connect(lambda a: self.custom_row.setVisible(a == "Custom"))
        ov.addSpacing(26)

        # AI model row (one click)
        ai = QHBoxLayout(); ai.addWidget(label("AI", "section")); ai.addStretch(1)
        self.scan_btn = button("Detect", "ghost", "down", "Find AI models installed on this PC", on=lambda: self.detect_local(True))
        ai.addWidget(self.scan_btn)
        ai.addWidget(button("Add", "ghost", "plus", "Add a cloud API connection", on=self.add_connection))
        ov.addLayout(ai); ov.addSpacing(8)
        self.conn_chips = Chips()
        self.conn_chips.changed.connect(self.select_connection)
        self.conn_chips.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conn_chips.customContextMenuRequested.connect(self._conn_menu)
        ov.addWidget(self.conn_chips)
        ov.addSpacing(10)
        mrow = QHBoxLayout(); mrow.setSpacing(8)
        mrow.addWidget(label("Model", "muted"))
        self.model_chips = Chips()
        self.model_chips.changed.connect(self.select_model)
        mrow.addWidget(self.model_chips, 1)
        self.more_models = QComboBox(); self.more_models.setMinimumWidth(180)
        self.more_models.setToolTip("All models from this provider")
        self.more_models.activated.connect(lambda i: self.select_model(self.more_models.itemData(i)))
        mrow.addWidget(self.more_models)
        self.refresh_models_btn = button("", "ghost", "down", "Reload model list", on=lambda: self.load_models(True))
        mrow.addWidget(self.refresh_models_btn)
        self.model_row = QWidget(); self.model_row.setLayout(mrow)
        ov.addWidget(self.model_row)
        self.ai_note = label("", "faint", wrap=True)
        ov.addWidget(self.ai_note)
        ov.addSpacing(10)
        tr_row = QHBoxLayout(); tr_row.setSpacing(8)
        tr_row.addWidget(label("Transcription", "muted"))
        self.whisper = Chips([(m, "Auto" if m == "auto" else m) for m in WHISPER_CHOICES[:5]] , wrap=False)
        tr_row.addWidget(self.whisper); tr_row.addStretch(1)
        ov.addLayout(tr_row)
        ov.addSpacing(24)

        # advanced
        self.adv_btn = button("Advanced", "ghost", "chevron", on=self.toggle_advanced)
        self.adv_btn.setStyleSheet("text-align: left; padding-left: 0;")
        ov.addWidget(self.adv_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.adv = QWidget()
        ag = QGridLayout(self.adv); ag.setContentsMargins(0, 14, 0, 0); ag.setHorizontalSpacing(36); ag.setVerticalSpacing(18)
        self.framing = Chips(FRAMING)
        self.quality = Chips(["1080p", "1440p", "4K"])
        self.device = Chips([("auto", "Auto"), ("cuda", "GPU"), ("cpu", "CPU")])
        self.language = QComboBox(); self.language.setEditable(True)
        for code, name in [("", "Detect automatically"), ("en", "English"), ("hi", "Hindi"), ("es", "Spanish"),
                           ("pt", "Portuguese"), ("ja", "Japanese"), ("ko", "Korean"), ("de", "German"),
                           ("fr", "French"), ("ru", "Russian"), ("ar", "Arabic"), ("id", "Indonesian")]:
            self.language.addItem(name, code)
        self.external = Chips([(True, "On"), (False, "Off")], wrap=False)
        self.scenes = Chips([(True, "On"), (False, "Off")], wrap=False)
        self.cap_quality = Chips([("best", "Best", "Listens again to each chosen clip with the most accurate model"),
                                  ("fast", "Fast", "Uses the first transcript")], wrap=False)
        out_row = QWidget(); orl = QHBoxLayout(out_row); orl.setContentsMargins(0, 0, 0, 0); orl.setSpacing(8)
        self.out_dir = QLineEdit(); self.out_dir.setPlaceholderText(str(shorts_dir()))
        orl.addWidget(self.out_dir, 1); orl.addWidget(button("Browse", on=self.browse_out))
        ag.addWidget(Field("Framing", self.framing), 0, 0)
        ag.addWidget(Field("Quality", self.quality), 0, 1)
        ag.addWidget(Field("Transcribe on", self.device), 1, 0)
        ag.addWidget(Field("Spoken language", self.language,
                           "Detect automatically works with Auto subtitles. If Hindi speech comes out in Urdu "
                           "script and you don't use AI, choose Hindi."), 1, 1)
        ag.addWidget(Field("Viewer signals", self.external,
                           "Most replayed, chat replay and timestamped comments when the link has them."), 2, 0)
        ag.addWidget(Field("Scene detection", self.scenes, "Cleaner cuts and better action detection."), 2, 1)
        ag.addWidget(Field("Caption accuracy", self.cap_quality,
                           "Best listens to each chosen clip again with Whisper large-v3. Adds about a minute "
                           "on a GPU, a few on a CPU."), 3, 0)
        ag.addWidget(Field("Save to", out_row), 4, 0, 1, 2)
        ag.setColumnStretch(0, 1); ag.setColumnStretch(1, 1)
        ov.addWidget(self.adv)
        self.adv.setVisible(False)
        ov.addSpacing(30)

        v.addWidget(self.options_box)

        # progress
        self.prog_panel = QFrame(); self.prog_panel.setObjectName("panel")
        pv = QVBoxLayout(self.prog_panel); pv.setContentsMargins(20, 18, 20, 18); pv.setSpacing(12)
        top = QHBoxLayout(); top.setSpacing(10)
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        self.prog_step = label("", "section"); tcol.addWidget(self.prog_step)
        self.prog_title = label("Starting", "h2"); tcol.addWidget(self.prog_title)
        top.addLayout(tcol); top.addStretch(1)
        self.prog_pct = label("", "h2"); top.addWidget(self.prog_pct, 0, Qt.AlignmentFlag.AlignBottom)
        self.cancel_btn = button("Cancel", "ghost", on=self.cancel); self.cancel_btn.setToolTip("Esc")
        top.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignBottom)
        pv.addLayout(top)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False)
        pv.addWidget(self.bar)
        self.steps = Steps(STAGE_SHORT); pv.addWidget(self.steps)
        drow = QHBoxLayout(); drow.setSpacing(12)
        self.prog_msg = label("", "muted", wrap=True); drow.addWidget(self.prog_msg, 1)
        self.prog_time = label("", "faint"); drow.addWidget(self.prog_time, 0, Qt.AlignmentFlag.AlignTop)
        pv.addLayout(drow)
        self.prog_note = label("", "faint", wrap=True); pv.addWidget(self.prog_note); self.prog_note.hide()
        v.addWidget(self.prog_panel)
        self.prog_panel.hide()
        self._tick = QTimer(self, interval=1000, timeout=self._update_time)

        # results
        self.results_head = QWidget()
        rh = QHBoxLayout(self.results_head); rh.setContentsMargins(0, 26, 0, 12)
        self.results_title = label("", "h2"); rh.addWidget(self.results_title); rh.addStretch(1)
        self.results_upload = button("Upload all", "ghost", "upload", "Upload these clips to YouTube")
        rh.addWidget(self.results_upload)
        self.results_open = button("Open folder", "ghost", "folder")
        rh.addWidget(self.results_open)
        v.addWidget(self.results_head)
        self.used_lbl = label("", "faint", wrap=True)
        v.addWidget(self.used_lbl)
        v.addSpacing(10)
        self.results = QWidget()
        self.results_grid = QGridLayout(self.results); self.results_grid.setContentsMargins(0, 0, 0, 0)
        self.results_grid.setSpacing(14)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v.addWidget(self.results)
        self.results_head.hide(); self.used_lbl.hide()
        v.addStretch(1)

        for chips in (self.count, self.length, self.aspect, self.captions, self.cap_lang, self.whisper, self.framing,
                      self.quality, self.device, self.external, self.scenes, self.cap_quality):
            chips.changed.connect(self._options_changed)
        self.create_scroll = scroll(page)
        return self.create_scroll

    # ---------- custom length
    def _len_min_changed(self, v: int) -> None:
        if v + 5 > self.len_max.value():
            self.len_max.blockSignals(True); self.len_max.setValue(min(600, v + 5)); self.len_max.blockSignals(False)
        self._len_hint(); self._options_changed()

    def _len_max_changed(self, v: int) -> None:
        if v - 5 < self.len_min.value():
            self.len_min.blockSignals(True); self.len_min.setValue(max(5, v - 5)); self.len_min.blockSignals(False)
        self._len_hint(); self._options_changed()

    def _len_hint(self) -> None:
        if self.length.value() != "custom":
            self.len_hint.setText("")
            return
        mx = self.len_max.value()
        self.len_hint.setText("Over 3 minutes won't count as a YouTube Short." if mx > 180 else "")

    def _set_length(self, mn: float, mx: float) -> None:
        mn, mx = int(round(float(mn))), int(round(float(mx)))
        if (mn, mx) in self.length.values:
            self.length.set_value((mn, mx))
        else:
            self.len_max.blockSignals(True); self.len_min.blockSignals(True)
            self.len_max.setValue(max(10, min(600, mx))); self.len_min.setValue(max(5, min(mn, self.len_max.value() - 5)))
            self.len_max.blockSignals(False); self.len_min.blockSignals(False)
            self.length.set_value("custom")
        self.len_row.setVisible(self.length.value() == "custom")
        self._len_hint()

    def _length_pair(self) -> tuple[float, float]:
        v = self.length.value()
        if v == "custom":
            return float(self.len_min.value()), float(self.len_max.value())
        return (float(v[0]), float(v[1])) if isinstance(v, tuple) else (20.0, 60.0)

    # ---------- folding the options while a job runs
    def _settings_summary(self) -> str:
        mn, mx = self._length_pair()
        fmt = lambda x: f"{int(x)}s" if x < 120 or x % 60 else f"{int(x // 60)} min"
        aspect = self.aspect.value() or "9:16"
        if aspect == "Custom":
            aspect = f"{self.cw.value()}×{self.ch.value()}"
        caps = self.captions.value() or "Bold"
        sub = dict((k, t) for k, t, *_ in SUB_MODES).get(self.cap_lang.value() or "auto", "Auto")
        c = self.conns.active
        ai = f"{c.model or c.name}" if c else "No AI"
        n = self.count.value() or 8
        parts = [f"{n} clip{'s' if n != 1 else ''}", f"{fmt(mn)}–{fmt(mx)}", aspect,
                 "No captions" if caps == "Off" else f"{caps} captions · {sub} subtitles", ai]
        return "   ·   ".join(parts)

    def _collapse_options(self, collapse: bool, scroll: bool = False) -> None:
        if collapse:
            self.summary_lbl.setText(self._settings_summary())
        self.options_box.setVisible(not collapse)
        self.summary_bar.setVisible(collapse)
        if scroll and not collapse:
            QTimer.singleShot(40, lambda: self._scroll_to(self.options_box, 120))

    def _scroll_to(self, widget: QWidget, above: int = 24) -> None:
        area = getattr(self, "create_scroll", None)
        if area is None or not widget.isVisible():
            return
        y = widget.mapTo(area.widget(), QPoint(0, 0)).y() - above
        bar = area.verticalScrollBar()
        target = max(0, min(bar.maximum(), y))
        self._scroll_anim = QPropertyAnimation(bar, b"value", self)
        self._scroll_anim.setDuration(380)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(target)
        self._scroll_anim.start()

    def toggle_advanced(self) -> None:
        on = not self.adv.isVisible()
        self.adv.setVisible(on)
        self.adv_btn.setIcon(icon("down" if on else "chevron"))
        self.qs.setValue("advanced_open", on)

    # ================= live =================
    def _live_page(self) -> QWidget:
        page, v = page_container()
        v.addWidget(label("Live", "h1")); v.addSpacing(4)
        v.addWidget(label("Watches a live stream and clips it when the streamer gets loud or chat speeds up.", "muted"))
        v.addSpacing(22)
        row = QHBoxLayout(); row.setSpacing(8)
        self.live_url = QLineEdit(); self.live_url.setObjectName("url")
        self.live_url.setPlaceholderText("Paste a live link  (Twitch, YouTube Live, Kick, ...)")
        self.live_url.setText(str(self.qs.value("live/url", "")))
        self.live_url.returnPressed.connect(self.toggle_live)
        row.addWidget(self.live_url, 1)
        self.live_btn = button("Start watching", "primary", on=self.toggle_live); self.live_btn.setMinimumHeight(38)
        row.addWidget(self.live_btn)
        v.addLayout(row)
        v.addSpacing(24)
        g = QGridLayout(); g.setHorizontalSpacing(36); g.setVerticalSpacing(20)
        self.live_sens = Chips([("Low", "Big moments only"), ("Normal", "Normal"), ("High", "More clips")])
        self.live_sens.set_value(str(self.qs.value("live/sens", "Normal")))
        self.live_finish = Chips([(True, "While watching", "Clips appear as they happen"),
                                  (False, "After the stream", "Keeps your GPU free while you play")])
        self.live_finish.set_value(str(self.qs.value("live/finish", "true")).lower() == "true")
        g.addWidget(Field("Sensitivity", self.live_sens), 0, 0)
        g.addWidget(Field("Make clips", self.live_finish), 0, 1)
        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)
        v.addLayout(g)
        v.addSpacing(8)
        v.addWidget(label("Format, captions, length, transcription and AI come from the Create page.", "faint"))
        v.addSpacing(26)

        self.live_panel = QFrame(); self.live_panel.setObjectName("panel")
        pv = QVBoxLayout(self.live_panel); pv.setContentsMargins(18, 16, 18, 16); pv.setSpacing(12)
        top = QHBoxLayout()
        self.live_state = label("Watching", "h2"); top.addWidget(self.live_state); top.addStretch(1)
        self.live_now = button("Clip now", tip="Clip the last 30 seconds (Space)", on=self._clip_now)
        top.addWidget(self.live_now)
        pv.addLayout(top)
        meters = QGridLayout(); meters.setHorizontalSpacing(18); meters.setVerticalSpacing(6)
        self.m_voice = QProgressBar(); self.m_voice.setRange(0, 100); self.m_voice.setTextVisible(False)
        self.m_voice.setObjectName("voice")
        self.m_chat = QProgressBar(); self.m_chat.setRange(0, 100); self.m_chat.setTextVisible(False)
        self.m_chat.setObjectName("chat")
        self.m_voice_lbl = label("", "faint"); self.m_chat_lbl = label("", "faint")
        meters.addWidget(label("Voice", "muted"), 0, 0); meters.addWidget(self.m_voice, 0, 1); meters.addWidget(self.m_voice_lbl, 0, 2)
        meters.addWidget(label("Chat", "muted"), 1, 0); meters.addWidget(self.m_chat, 1, 1); meters.addWidget(self.m_chat_lbl, 1, 2)
        meters.setColumnStretch(1, 1)
        pv.addLayout(meters)
        self.live_info = label("", "faint", wrap=True); pv.addWidget(self.live_info)
        v.addWidget(self.live_panel); self.live_panel.hide()

        self.live_head = QWidget(); lh = QHBoxLayout(self.live_head); lh.setContentsMargins(0, 26, 0, 12)
        self.live_count = label("", "h2"); lh.addWidget(self.live_count); lh.addStretch(1)
        self.live_open = button("Open folder", "ghost", "folder"); lh.addWidget(self.live_open)
        v.addWidget(self.live_head); self.live_head.hide()
        from .widgets import FlowLayout
        self.live_grid_w = QWidget(); self.live_grid = FlowLayout(self.live_grid_w, spacing=14)
        v.addWidget(self.live_grid_w)
        v.addSpacing(20)
        v.addWidget(label("Only clip streams you're allowed to repost. Expect a delay of a few seconds behind live.",
                          "faint", wrap=True))
        v.addStretch(1)
        QShortcut(QKeySequence("Space"), page, activated=self._clip_now)
        return scroll(page)

    def toggle_live(self) -> None:
        if self.live:
            self.stop_live()
            return
        url = self.live_url.text().strip()
        if not sources.is_url(url):
            self.toast.show_text("Paste a live stream link first.", error=True)
            return
        if self.worker:
            self.toast.show_text("Wait for the current job to finish.", error=True)
            return
        from ..core.live import LiveOptions
        cfg = self._config(); cfg.source = url
        opts = LiveOptions(self.live_sens.value() or "Normal", self.live_finish.value() is not False)
        self.qs.setValue("live/url", url); self.qs.setValue("live/sens", opts.sensitivity)
        self.qs.setValue("live/finish", opts.finish_now)
        self._remember_root(cfg.output_dir)
        while self.live_grid.count():
            it = self.live_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.live_head.hide()
        self.live_panel.show()
        self.live_state.setText("Connecting"); self.live_info.setText("")
        self.live_btn.setText("Stop"); self.live_btn.setObjectName("")
        self.live_btn.style().unpolish(self.live_btn); self.live_btn.style().polish(self.live_btn)
        self.live_url.setEnabled(False); self.gen_btn.setEnabled(False)
        self.live_count_n = 0
        key = self.session_keys.get(cfg.connection_id, "")
        self.live = workers.LiveWorker(cfg, opts, self.conns, key)
        self.live.status.connect(self._live_status)
        self.live.clip.connect(self._live_clip)
        self.live.note.connect(lambda s: self.live_info.setText(s))
        self.live.finished.connect(self._live_done)
        self.live.failed.connect(self._live_failed)
        workers.start(self, self.live, self.live.finished, self.live.failed)

    def stop_live(self) -> None:
        if not self.live:
            return
        self.live.session.stop.set()
        self.live_btn.setEnabled(False)
        self.live_state.setText("Finishing clips")

    def _clip_now(self) -> None:
        if self.live and self.stack.currentIndex() == 1:
            self.live.session.request_clip()
            self.toast.show_text("Clipping the last 30 seconds")

    def _live_status(self, s) -> None:
        names = {"Watching": "Watching", "Warming up": "Learning this stream's normal level",
                 "Connecting": "Connecting", "Finishing": "Finishing clips", "Stopped": "Stopped"}
        state = names.get(s.state, s.state)
        if s.in_moment and s.state == "Watching":
            state = dot_text(theme.RED, "Recording a moment")
        elif s.state == "Watching":
            state = dot_text(theme.TEAL, state)
        self.live_state.setText(state)
        self.m_voice.setValue(int(min(100, s.voice / 4 * 100)))
        self.m_voice_lbl.setText("loud" if s.voice > 2 else "normal")
        if s.chat_source:
            self.m_chat.setValue(int(min(100, s.chat_rate * 4)))
            self.m_chat_lbl.setText(f"{s.chat_rate / 5:.1f}/s")
        else:
            self.m_chat.setValue(0); self.m_chat_lbl.setText("not available")
        rec = int(s.recorded)
        parts = [f"Live for {rec // 3600}:{rec % 3600 // 60:02d}:{rec % 60:02d}", f"{s.made} clips"]
        if s.queued:
            parts.append(f"{s.queued} being made")
        if s.note:
            parts.append(s.note)
        self.live_info.setText("  ·  ".join(parts))
        self.live_now.setEnabled(s.state in ("Watching", "Warming up"))

    def _live_clip(self, c: dict, folder: str) -> None:
        self.live_count_n += 1
        self.live_head.show()
        self.live_count.setText(f"{self.live_count_n} clips")
        try:
            self.live_open.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.live_open.clicked.connect(lambda: open_path(folder))
        self.live_grid.addWidget(ClipCard(c, folder, self.toast.show_text))
        self.live_grid_w.updateGeometry()

    def _live_reset(self) -> None:
        self.live = None
        self.live_btn.setEnabled(True); self.live_btn.setText("Start watching"); self.live_btn.setObjectName("primary")
        self.live_btn.style().unpolish(self.live_btn); self.live_btn.style().polish(self.live_btn)
        self.live_url.setEnabled(True); self.gen_btn.setEnabled(True)
        self.live_now.setEnabled(False)

    def _live_done(self) -> None:
        n = getattr(self, "live_count_n", 0)
        self._live_reset()
        self.live_state.setText(f"Stopped  ·  {n} clips")
        QApplication.alert(self)

    def _live_failed(self, msg: str) -> None:
        self._live_reset()
        self.live_state.setText("Couldn't watch this stream")
        self.live_info.setText(msg)

    def _upload(self, clips: list[dict]) -> None:
        if not self.studio.store.channels:
            self._go(3)
            self.toast.show_text("Connect your YouTube channel first.", 4000, error=True)
            return
        self.studio.upload_clips(clips)

    # ================= library =================
    def _library_page(self) -> QWidget:
        page, v = page_container()
        head = QHBoxLayout(); head.addWidget(label("Library", "h1")); head.addStretch(1)
        head.addWidget(button("Open folder", "ghost", "folder", on=lambda: open_path(str(shorts_dir()))))
        v.addLayout(head); v.addSpacing(4)
        self.lib_count = label("Everything you've made, newest first.", "muted")
        v.addWidget(self.lib_count)
        v.addSpacing(18)
        self.lib_search = QLineEdit(); self.lib_search.setPlaceholderText("Search titles, hashtags, links...")
        self.lib_search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self, singleShot=True, interval=250, timeout=lambda: self._render_library(reset=True))
        self.lib_search.textChanged.connect(lambda _: self._search_timer.start())
        v.addWidget(self.lib_search)
        v.addSpacing(22)
        self.lib_box = QVBoxLayout(); self.lib_box.setSpacing(30)
        v.addLayout(self.lib_box)
        self.lib_more = button("Show more", on=lambda: self._render_library(reset=False))
        v.addSpacing(16)
        v.addWidget(self.lib_more, 0, Qt.AlignmentFlag.AlignLeft)
        v.addStretch(1)
        self.lib_items: list[dict] = []
        self.lib_shown = 0
        return scroll(page)

    def _library_roots(self) -> list[str]:
        r = self.qs.value("library_roots", []) or []
        return [r] if isinstance(r, str) else list(r)

    def _remember_root(self, folder: str) -> None:
        if not folder:
            return
        roots = self._library_roots()
        if folder not in roots:
            self.qs.setValue("library_roots", roots + [folder])

    def _load_library(self) -> None:
        self.lib_items = engine.read_library(self._library_roots())
        self._render_library(reset=True)

    def _render_library(self, reset: bool) -> None:
        from .widgets import FlowLayout
        if reset:
            while self.lib_box.count():
                it = self.lib_box.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            self.lib_shown = 0
        q = self.lib_search.text()
        items = [d for d in self.lib_items if engine.matches(d, q)]
        total_clips = sum(len(d.get("clips", [])) for d in self.lib_items)
        self.lib_count.setText(f"{len(self.lib_items)} videos  ·  {total_clips} clips" if self.lib_items
                               else "Everything you've made, newest first.")
        if not items:
            self.lib_box.addWidget(label("No matches." if q.strip() else
                                         "Nothing here yet. Clips you generate will show up here.", "muted"))
            self.lib_more.hide()
            return
        page = items[self.lib_shown:self.lib_shown + 20]
        for d in page:
            block = QWidget(); bv = QVBoxLayout(block); bv.setContentsMargins(0, 0, 0, 0); bv.setSpacing(12)
            hh = QHBoxLayout(); hh.setSpacing(6)
            t = label(d.get("title") or "Untitled", "h2"); t.setToolTip(d.get("source", ""))
            hh.addWidget(t)
            hh.addSpacing(6)
            hh.addWidget(label(f"{d.get('created', '')}  ·  {len(d.get('clips', []))} clips", "faint"))
            hh.addStretch(1)
            lib_ups = [{"path": os.path.join(d["folder"], c.get("file", "")), "title": c.get("title", ""),
                        "description": c.get("description", ""), "hashtags": c.get("hashtags") or []}
                       for c in sorted(d.get("clips", []), key=lambda c: -(c.get("score") or 0))]
            hh.addWidget(button("Upload all", "ghost", tip="Upload these clips to YouTube",
                                on=lambda _, u=lib_ups: self._upload(u)))
            hh.addWidget(button("Clip again", "ghost", tip="Load this video and its settings",
                                on=lambda _, d=d: self.reclip(d)))
            hh.addWidget(button("", "ghost", "folder", "Open folder", on=lambda _, f=d["folder"]: open_path(f)))
            hh.addWidget(button("Delete", "ghost", tip="Delete these clips from disk",
                                on=lambda _, d=d, b=block: self.delete_job(d, b)))
            bv.addLayout(hh)
            grid = QWidget(); fl = FlowLayout(grid, spacing=14)
            clips = sorted(d.get("clips", []), key=lambda c: -(c.get("score") or 0))
            for c in clips:
                fl.addWidget(ClipCard(c, d["folder"], self.toast.show_text))
            bv.addWidget(grid)
            self.lib_box.addWidget(block)
        self.lib_shown += len(page)
        self.lib_more.setVisible(self.lib_shown < len(items))

    def reclip(self, d: dict) -> None:
        src = d.get("source", "")
        if not src or not (sources.is_url(src) or os.path.isfile(src)):
            self.toast.show_text("The original video is no longer there.", error=True)
            return
        s = d.get("settings", {})
        self._loading = True
        pairs = {"clip_count": self.count, "aspect": self.aspect, "captions": self.captions, "caption_lang": self.cap_lang,
                 "framing": self.framing, "quality": self.quality, "whisper_model": self.whisper,
                 "device": self.device, "use_external": self.external, "use_scenes": self.scenes,
                 "caption_quality": self.cap_quality}
        for k, chips in pairs.items():
            if k in s:
                chips.set_value(min(10, max(1, int(s[k]))) if k == "clip_count" else s[k])
        if "min_len" in s and "max_len" in s:
            self._set_length(s["min_len"], s["max_len"])
        if s.get("aspect") == "Custom":
            self.cw.setValue(int(s.get("custom_w", 1080))); self.ch.setValue(int(s.get("custom_h", 1920)))
        self.custom_row.setVisible(self.aspect.value() == "Custom")
        i = self.language.findData(s.get("language", ""))
        self.language.setCurrentIndex(max(0, i))
        self.out_dir.setText(s.get("output_dir", ""))
        cid = s.get("connection_id", "")
        if cid == "" or self.conns.get(cid):
            self.conns.set_active(cid)
            c = self.conns.get(cid)
            if c and s.get("llm_model"):
                c.model = s["llm_model"]; self.conns.upsert(c)
            self._refresh_connections()
        self._loading = False
        self.preset_chips.set_value(None)
        self.source = ""
        self._set_source(src)
        self._save()
        self._go(0)
        self.toast.show_text("Loaded. Change anything, then Generate.")

    def delete_job(self, d: dict, block: QWidget) -> None:
        n = len(d.get("clips", []))
        if QMessageBox.question(self, "Delete clips",
                                f"Delete {n} clips of \"{d.get('title', 'this video')}\" from your PC?\n"
                                "The original video is not touched.") != QMessageBox.StandardButton.Yes:
            return
        try:
            engine.delete_job(d["folder"])
        except Exception as exc:
            self.toast.show_text(f"Couldn't delete: {exc}", error=True)
            return
        self.lib_items = [x for x in self.lib_items if x is not d]
        block.deleteLater()
        self.lib_shown = max(0, self.lib_shown - 1)
        self._render_library(reset=True)
        self.toast.show_text("Deleted")

    # ================= settings =================
    def _settings_page(self) -> QWidget:
        page, v = page_container()
        v.addWidget(label("Settings", "h1")); v.addSpacing(28)
        v.addWidget(label("APPEARANCE", "section")); v.addSpacing(10)
        self.accent_chips = Chips([(hx, name) for name, hx in theme.ACCENTS], wrap=False)
        self.accent_chips.set_value(theme.ACCENT)
        for k, (name, hx) in enumerate(theme.ACCENTS):
            btn = self.accent_chips.group.button(k)
            if btn is not None:
                btn.setIcon(_swatch(hx)); btn.setIconSize(QSize(10, 10))
        self.accent_chips.changed.connect(self.apply_accent)
        v.addWidget(Field("Accent color", self.accent_chips, "Used for buttons, selections, progress and the active page."))
        v.addSpacing(32)
        v.addWidget(label("CONNECTIONS", "section")); v.addSpacing(10)
        v.addWidget(label("AI providers you've added. Click one on the Create page to switch.", "muted"))
        v.addSpacing(12)
        self.conn_list = QVBoxLayout(); self.conn_list.setSpacing(8)
        v.addLayout(self.conn_list)
        v.addSpacing(10)
        brow = QHBoxLayout(); brow.setSpacing(8)
        brow.addWidget(button("Detect local models", ic="down", on=lambda: self.detect_local(True)))
        brow.addWidget(button("Add cloud API", ic="plus", on=self.add_connection)); brow.addStretch(1)
        v.addLayout(brow)
        v.addSpacing(6)
        v.addWidget(label("Local apps are found automatically: Ollama, LM Studio, Jan, llama.cpp, KoboldCpp, vLLM. "
                          "Checked at start and every minute.", "faint", wrap=True))
        v.addSpacing(32)
        v.addWidget(label("SYSTEM", "section")); v.addSpacing(10)
        self.sys_lbl = label("", "muted", wrap=True)
        self.sys_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.sys_lbl)
        v.addSpacing(32)
        v.addWidget(label("SHORTCUTS", "section")); v.addSpacing(10)
        v.addWidget(label("Ctrl+O  open file     Ctrl+L  paste link     Ctrl+Enter  generate     Esc  cancel     "
                          "Ctrl+1–5  switch page     Space  clip now (Live)", "muted", wrap=True))
        v.addStretch(1)
        return scroll(page)

    def apply_accent(self, hx: str) -> None:
        """Change the accent everywhere, right away."""
        hx = theme.set_accent(hx)
        self.qs.setValue("accent", hx)
        app = QApplication.instance()
        app.setStyleSheet(theme.build_qss(hx))
        self.setWindowIcon(app_icon()); app.setWindowIcon(app_icon())
        self.brand_icon.setPixmap(app_icon().pixmap(22, 22))
        for b, name in zip(self.nav, self.nav_icons):
            b.setIcon(icon(name, on_color=theme.ACCENT))
        if hasattr(self, "toast"):
            self.toast._style(False)
        for wdg in app.allWidgets():
            wdg.update()

    def _refresh_settings_list(self) -> None:
        while self.conn_list.count():
            it = self.conn_list.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for c in self.conns.items:
            f = QFrame(); f.setObjectName("panel")
            h = QHBoxLayout(f); h.setContentsMargins(14, 10, 10, 10); h.setSpacing(10)
            col = QVBoxLayout(); col.setSpacing(2)
            col.addWidget(label(c.name + ("   ·  active" if c.id == self.conns.active_id else "")))
            has = bool(self.conns.api_key(c)) or not c.needs_key
            extra = f"{len(c.models_cache)} models found" if c.is_local else ("key saved" if has else "no key")
            col.addWidget(label(f"{c.provider}  ·  {c.model or 'no model'}  ·  {extra}", "faint"))
            h.addLayout(col, 1)
            h.addWidget(button("Edit", "ghost", on=lambda _, c=c: self.edit_connection(c)))
            h.addWidget(button("Remove", "ghost", on=lambda _, c=c: self.remove_connection(c)))
            self.conn_list.addWidget(f)

    # ================= environment =================
    def _check_env(self) -> None:
        ok, name, vram = cuda_info()
        dev = f"{name} · {vram // 1024} GB" if ok and vram else (name if ok else "No GPU found, using CPU")
        self.device_lbl.setText(dev)
        lines = [f"GPU: {dev}"]
        if ok:
            from ..core.gpu import gpu_failure, missing_cuda_libs
            try:
                miss = missing_cuda_libs()
            except Exception:
                miss = []
            if gpu_failure():
                lines.append(f"GPU transcription: not working. {gpu_failure()}")
            elif miss:
                lines.append(f"GPU transcription: libraries missing ({', '.join(miss[:2])}). Restart run.bat to install them.")
            else:
                lines.append("GPU transcription: ready")
        try:
            from ..core.ffmpeg import ffmpeg_path, nvenc_ok
            lines.append(f"FFmpeg: {ffmpeg_path()}")
            lines.append(f"GPU encoding: {'available' if nvenc_ok() else 'not available, using CPU'}")
        except Exception as exc:
            lines.append(f"FFmpeg: {exc}")
        from ..core.transcribe import available as wa
        lines.append(f"Whisper: {'installed' if wa() else 'not installed (pip install faster-whisper)'}")
        lines.append(f"Key storage: {'Windows Credential Manager' if self.conns.keys.secure else 'session only'}")
        lines.append(f"Clips folder: {shorts_dir()}")
        self.sys_lbl.setText("\n".join(lines))

    # ================= source =================
    def browse(self) -> None:
        start = self.qs.value("last_dir", str(Path.home() / "Videos"))
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", start,
                                              "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.flv *.ts *.wmv);;All files (*.*)")
        if path:
            self.qs.setValue("last_dir", os.path.dirname(path))
            self._set_source(path)

    def browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save clips to", self.out_dir.text() or str(shorts_dir()))
        if d:
            self.out_dir.setText(d)

    def _set_source(self, text: str) -> None:
        text = (text or "").strip().strip('"')
        if not text or text == self.source:
            return
        if sources.is_url(text):
            self.url.setText(text)
            self.drop.set_file("Link", "Reading...")
        elif os.path.isfile(text):
            if not sources.is_video_file(text):
                self.toast.show_text("That doesn't look like a video file.", error=True)
                return
            self.url.clear()
            self.drop.set_file(os.path.basename(text), "Reading...")
        else:
            self.toast.show_text("Couldn't find that file or link.", error=True)
            return
        self.source = text
        w = workers.ProbeWorker(text)
        w.done.connect(self._probed)
        workers.start(self, w, w.done)

    def _probed(self, src: str, title: str, sub: str) -> None:
        if src != self.source:
            return
        self.drop.set_file(title or os.path.basename(src) or src, sub)

    # app-wide drag & drop
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        md = e.mimeData()
        if md.hasUrls() or md.hasText():
            e.acceptProposedAction()
            self._go(0)
            self.drop.set_active(True)

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self.drop.set_active(False)

    def dropEvent(self, e) -> None:  # noqa: N802
        self.drop.set_active(False)
        md = e.mimeData()
        if md.hasUrls():
            u = md.urls()[0]
            self._set_source(u.toLocalFile() if u.isLocalFile() else u.toString())
        elif md.hasText():
            self._set_source(md.text())

    # ================= presets =================
    def _refresh_presets(self) -> None:
        self.preset_chips.set_options([(p.name, p.name) for p in self.presets.all], keep=False)
        cur = self.qs.value("preset", "")
        if cur and self.presets.find(cur):
            self.preset_chips.set_value(cur)

    def apply_preset(self, name: str) -> None:
        p = self.presets.find(name)
        if not p:
            return
        self._loading = True
        v = p.values
        setters = {"whisper_model": self.whisper, "device": self.device, "quality": self.quality,
                   "use_external": self.external, "use_scenes": self.scenes, "framing": self.framing,
                   "aspect": self.aspect, "captions": self.captions, "caption_lang": self.cap_lang,
                   "caption_quality": self.cap_quality,
                   "clip_count": self.count}
        for k, chips in setters.items():
            if k in v:
                chips.set_value(min(10, max(1, int(v[k]))) if k == "clip_count" else v[k])
        if "min_len" in v and "max_len" in v:
            self._set_length(v["min_len"], v["max_len"])
        cid = resolve_llm_for_preset(p, self.conns)
        if cid is not None:
            self.conns.set_active(cid)
            self.conn_chips.set_value(cid)
            self._update_model_row()
        self._loading = False
        self.qs.setValue("preset", name)
        self._save()
        self.toast.show_text(f"{name} applied")

    def save_preset(self) -> None:
        name = ask_text(self, "Save preset", "Name")
        if not name:
            return
        cfg = self._config()
        vals = asdict(cfg) | {"use_llm": bool(cfg.connection_id)}
        p = self.presets.save_custom(name, vals)
        self._refresh_presets()
        self.preset_chips.set_value(p.name)
        self.qs.setValue("preset", p.name)
        self.toast.show_text(f"Saved {p.name}")

    def _preset_menu(self, pos) -> None:
        btn = self.preset_chips.childAt(pos)
        if not isinstance(btn, QPushButton):
            return
        name = btn.text()
        p = self.presets.find(name)
        if not p or p.builtin:
            return
        m = QMenu(self)
        m.addAction("Delete preset", lambda: (self.presets.delete(name), self._refresh_presets()))
        m.exec(self.preset_chips.mapToGlobal(pos))

    def _options_changed(self, *_) -> None:
        if self._loading:
            return
        self.preset_chips.set_value(None)
        self.qs.setValue("preset", "")
        self._save()

    # ================= connections & models =================
    def _refresh_connections(self) -> None:
        opts = [("", "None", "Use signals only, no AI")]
        for c in self.conns.items:
            ok = (not c.needs_key) or bool(self.conns.api_key(c)) or c.id in self.session_keys
            if c.is_local:
                tip = f"{len(c.models_cache)} models on this PC" if c.models_cache else "Not running"
            else:
                tip = c.provider + ("" if ok else "  ·  no key yet")
            opts.append((c.id, c.name, tip))
        self.conn_chips.set_options(opts, keep=False)
        self.conn_chips.set_value(self.conns.active_id if self.conns.active else "")
        self._update_model_row()
        self._refresh_settings_list()

    def detect_local(self, manual: bool, quiet: bool = False) -> None:
        if getattr(self, "_detecting", False) or self.worker or self.live:
            return
        self._detecting = True
        self._detect_manual, self._detect_quiet = manual, quiet
        self.scan_btn.setEnabled(False)
        if manual:
            self.scan_btn.setText("Detecting...")
        w = workers.DetectWorker()
        w.done.connect(self._detected)
        workers.start(self, w, w.done)

    def _detected(self, found: list, notes: list) -> None:
        self._detecting = False
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Detect")
        before = {c.id: list(c.models_cache) for c in self.conns.items}
        added, _ = self.conns.merge_detected(found)
        changed = bool(added) or any(before.get(c.id) != c.models_cache for c in self.conns.items)
        if changed or self._detect_manual:
            self._refresh_connections()
        n = sum(len(c.models_cache) for c in found)
        if self._detect_manual:
            msg = (f"Found {n} model{'s' if n != 1 else ''} on this PC" if n else (notes[-1] if notes else "Nothing found"))
            self.toast.show_text(msg, error=not n)
        elif added and not self._detect_quiet:
            self.toast.show_text(f"Found {n} local model{'s' if n != 1 else ''}")
        if not found and not self._detect_quiet and not any(c.models_cache for c in self.conns.items):
            self.ai_note.setText(notes[-1] if notes else "")

    def select_connection(self, cid: str) -> None:
        self.conns.set_active(cid)
        self._update_model_row()
        self._refresh_settings_list()
        self.preset_chips.set_value(None)

    def _update_model_row(self) -> None:
        c = self.conns.active
        self.model_row.setVisible(bool(c))
        if not c:
            self.ai_note.setText("Without AI, clips are picked from audio, speech and viewer signals.")
            return
        if c.needs_key and not self.conns.api_key(c):
            self.ai_note.setText(f"{c.name} has no API key. Right-click it and choose Edit to add one.")
        elif c.is_local and not c.models_cache:
            self.ai_note.setText(f"{c.provider} has no models yet. Download one, then click Detect.")
        else:
            self.ai_note.setText("")
        self._fill_models(c, c.models_cache)
        if not c.models_cache and not c.is_local:
            self.load_models(False)

    def _fill_models(self, c, models: list[str]) -> None:
        if c.is_local:
            quick = list(models[:8])           # everything you have, one click each
            if c.model and c.model not in quick:
                quick = [c.model] + quick[:7]
        else:
            rec = [m for m in c.recommended if not models or m in models]
            quick = rec[:3] if rec else models[:3]
            if c.model and c.model not in quick:
                quick = [c.model] + quick[:2]
        info = getattr(c, "model_info", {}) or {}
        self.model_chips.set_options([(m, m, info.get(m) or ("Recommended" if m in c.recommended else ""))
                                      for m in quick], keep=False)
        self.model_chips.set_value(c.model)
        self.more_models.blockSignals(True)
        self.more_models.clear()
        self.more_models.addItem(f"All models ({len(models)})" if models else "No list yet", None)
        for m in models:
            extra = (c.model_info or {}).get(m, "")
            self.more_models.addItem(("★ " if m in c.recommended else "    ") + m + (f"   {extra}" if extra else ""), m)
        self.more_models.blockSignals(False)

    def load_models(self, force: bool) -> None:
        c = self.conns.active
        if not c:
            return
        if c.models_cache and not force:
            return
        key = self.session_keys.get(c.id) or self.conns.api_key(c)
        if c.needs_key and not key:
            return
        self.refresh_models_btn.setEnabled(False)
        w = workers.ModelsWorker(c, key)
        w.done.connect(self._models_loaded)
        workers.start(self, w, w.done)

    def _models_loaded(self, cid: str, models: list, err: str) -> None:
        self.refresh_models_btn.setEnabled(True)
        c = self.conns.get(cid)
        if not c:
            return
        if err:
            if cid == self.conns.active_id:
                hint = f" Is {c.provider} running?" if c.is_local else ""
                self.ai_note.setText(f"Couldn't load models from {c.name}.{hint}")
            return
        c.models_cache = models
        if not c.model and models:
            c.model = next((m for m in c.recommended if m in models), models[0])
        self.conns.upsert(c)
        if cid == self.conns.active_id:
            self._fill_models(c, models)

    def select_model(self, m) -> None:
        c = self.conns.active
        if not c or not m:
            return
        c.model = m
        self.conns.upsert(c)
        self._fill_models(c, c.models_cache)
        self._refresh_settings_list()

    def _conn_menu(self, pos) -> None:
        btn = self.conn_chips.childAt(pos)
        if not isinstance(btn, QPushButton):
            return
        i = self.conn_chips.group.id(btn)
        cid = self.conn_chips.values[i] if i >= 0 else ""
        c = self.conns.get(cid)
        if not c:
            return
        m = QMenu(self)
        m.addAction("Edit", lambda: self.edit_connection(c))
        m.addAction("Remove", lambda: self.remove_connection(c))
        m.exec(self.conn_chips.mapToGlobal(pos))

    def add_connection(self) -> None:
        d = ConnectionDialog(self, self.conns)
        if d.exec() and d.result_conn:
            self.conns.set_active(d.result_conn.id)
            self._refresh_connections()
            self.load_models(True)

    def edit_connection(self, c) -> None:
        d = ConnectionDialog(self, self.conns, c)
        if d.exec():
            self._refresh_connections()
            self.load_models(True)

    def remove_connection(self, c) -> None:
        if QMessageBox.question(self, "Remove connection", f"Remove {c.name}? Its saved key is deleted too.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.conns.remove(c.id)
        self._refresh_connections()

    # ================= job =================
    def _config(self) -> JobConfig:
        ln = self._length_pair()
        c = self.conns.active
        return JobConfig(
            source=self.source, clip_count=self.count.value() or 8, min_len=float(ln[0]), max_len=float(ln[1]),
            aspect=self.aspect.value() or "9:16", custom_w=self.cw.value() // 2 * 2, custom_h=self.ch.value() // 2 * 2,
            framing=self.framing.value() or "Auto", quality=self.quality.value() or "1080p",
            captions=self.captions.value() or "Bold", caption_lang=self.cap_lang.value() or "auto", whisper_model=self.whisper.value() or "auto",
            device=self.device.value() or "auto", language=self.language.currentData() or "",
            connection_id=c.id if c else "", llm_model=c.model if c else "",
            use_external=self.external.value() is not False, use_scenes=self.scenes.value() is not False,
            caption_quality=self.cap_quality.value() or "best",
            output_dir=self.out_dir.text().strip())

    def generate(self) -> None:
        if self.worker or self.live:
            return
        if self.url.text().strip() and self.url.text().strip() != self.source:
            self._set_source(self.url.text())
        if not self.source:
            self.toast.show_text("Add a video or paste a link first.", error=True)
            return
        cfg = self._config()
        self._remember_root(cfg.output_dir)
        self._save()
        self._clear_results()
        self.prog_panel.show()
        self.steps.reset(); self.steps.set_current(0)
        self.prog_step.setText(f"STEP 1 OF {len(STAGE_NAMES)}")
        self.prog_title.setText(STAGE_LONG["Import"])
        self.bar.setValue(0); self.prog_pct.setText("0%"); self.prog_msg.setText("Starting...")
        self.prog_note.clear(); self.prog_note.hide(); self.prog_time.setText("")
        self.cancel_btn.show(); self.cancel_btn.setEnabled(True)
        self._t0 = time.monotonic(); self._tick.start(); self._update_time()
        self._collapse_options(True)
        QTimer.singleShot(60, lambda: self._scroll_to(self.prog_panel, 24))
        self.gen_btn.setEnabled(False); self.gen_btn.setText("Generating...")
        key = self.session_keys.get(cfg.connection_id, "")
        self._job_token = getattr(self, "_job_token", 0) + 1
        self.worker = workers.JobWorker(cfg, self.conns, key, self._job_token)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        workers.start(self, self.worker, self.worker.finished, self.worker.failed, self.worker.cancelled)

    def cancel(self) -> None:
        if not self.worker:
            return
        self.worker.cancel.set()
        self.cancel_btn.setEnabled(False)
        self.prog_step.setText("CANCELLING")
        self.prog_title.setText("Stopping")
        self.prog_msg.setText("Stopping the current step...")
        token = self._job_token
        QTimer.singleShot(2500, lambda: self._force_stop(token))

    def _force_stop(self, token: int) -> None:
        """The job didn't stop within 2.5 s (it's inside a long step). Free the UI now;
        the old job stops on its own in the background and anything it sends is ignored."""
        if self.worker is None or token != self._job_token:
            return
        self._job_token += 1
        self._cancelled(force=True)

    def _is_current(self, token: int) -> bool:
        return self.worker is not None and token == getattr(self, "_job_token", None)

    def _progress(self, token: int, stage: str, frac: float, msg: str) -> None:
        if not self._is_current(token) or self.worker.cancel.is_set():
            return
        if frac < 0:                                   # a note / warning, not progress
            self.prog_note.setText(msg); self.prog_note.show()
            return
        pct = engine.overall(stage, frac)
        if pct >= self.bar.value():
            self.bar.setValue(pct); self.prog_pct.setText(f"{pct}%")
        if stage in STAGE_NAMES:
            i = STAGE_NAMES.index(stage)
            if i > self.steps.current:
                self.steps.set_current(i)
            if msg.startswith("Skipped"):
                self.steps.skip(i)
            self.prog_step.setText(f"STEP {i + 1} OF {len(STAGE_NAMES)}")
            self.prog_title.setText(STAGE_LONG.get(stage, stage))
        if msg:
            self.prog_msg.setText(msg)

    def _update_time(self) -> None:
        el = time.monotonic() - getattr(self, "_t0", time.monotonic())
        txt = _clock(el)
        pct = self.bar.value()
        if 6 <= pct < 100 and el > 20:
            left = el / pct * (100 - pct)
            txt += f"  ·  about {_clock_left(left)} left"
        self.prog_time.setText(txt)

    def _job_done(self) -> None:
        self.worker = None
        self._tick.stop()
        self.gen_btn.setEnabled(True); self.gen_btn.setText("Generate clips")
        self.cancel_btn.setEnabled(False)

    def _finished(self, token: int, res) -> None:
        if not self._is_current(token):
            return
        self._job_done()
        took = time.monotonic() - getattr(self, "_t0", time.monotonic())
        self.bar.setValue(100); self.prog_pct.setText("100%")
        self.steps.set_current(len(STAGE_NAMES))
        self.prog_step.setText("DONE")
        self.prog_title.setText(f"{len(res.clips)} clips ready")
        self.prog_msg.setText("Scroll down to watch, copy or upload them.")
        self.prog_time.setText(f"took {_clock(took)}")
        self.cancel_btn.hide()
        clips = [{"file": c.file, "thumb": c.thumb, "start": c.start, "end": c.end, "score": c.score,
                  "signal_score": c.signal_score, "llm_score": c.llm_score, "title": c.title, "reason": c.reason,
                  "description": c.description, "hashtags": c.hashtags} for c in res.clips]
        self._show_results(res.title, res.folder, clips, res.used, res.warnings)
        QTimer.singleShot(80, lambda: self._scroll_to(self.prog_panel, 24))
        QApplication.alert(self)

    def _failed(self, token: int, msg: str) -> None:
        if not self._is_current(token):
            return
        self._job_done()
        i = max(0, self.steps.current)
        self.steps.failed = True; self.steps.update()
        self.prog_step.setText(f"STOPPED AT STEP {i + 1} OF {len(STAGE_NAMES)}")
        self.prog_title.setText("Something went wrong")
        self.prog_msg.setText(msg)
        self.prog_msg.setObjectName("error"); self.prog_msg.style().unpolish(self.prog_msg); self.prog_msg.style().polish(self.prog_msg)
        self.cancel_btn.hide()

    def _cancelled(self, token: int = -1, force: bool = False) -> None:
        if not force and not self._is_current(token):
            return
        self._job_done()
        self.prog_panel.hide()
        self._collapse_options(False)
        self.toast.show_text("Cancelled")

    def _clear_results(self) -> None:
        self.prog_msg.setObjectName("muted"); self.prog_msg.style().unpolish(self.prog_msg); self.prog_msg.style().polish(self.prog_msg)
        while self.results_grid.count():
            it = self.results_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.results_head.hide(); self.used_lbl.hide()

    def _show_results(self, title: str, folder: str, clips: list[dict], used: list[str], warnings: list[str]) -> None:
        self.results_title.setText(f"{len(clips)} clips  ·  {title}")
        try:
            self.results_open.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.results_open.clicked.connect(lambda: open_path(folder))
        try:
            self.results_upload.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        ups = [{"path": c["file"] if os.path.isabs(c["file"]) else os.path.join(folder, c["file"]),
                "title": c.get("title", ""), "description": c.get("description", ""),
                "hashtags": c.get("hashtags") or []} for c in sorted(clips, key=lambda c: -(c.get("score") or 0))]
        self.results_upload.clicked.connect(lambda: self._upload(ups))
        self.results_head.show()
        note = "Used: " + ", ".join(used) if used else ""
        warnings = [w for w in warnings if w != "captions_dropped"]
        if warnings:
            note += ("\n" if note else "") + "  ".join(warnings[:3])
        self.used_lbl.setText(note); self.used_lbl.setVisible(bool(note))
        cols = max(1, (self.stack.width() - 80) // 240)
        for i, c in enumerate(sorted(clips, key=lambda c: -(c.get("score") or 0))):
            self.results_grid.addWidget(ClipCard(c, folder, self.toast.show_text), i // cols, i % cols)

    # ================= persistence =================
    def _save(self) -> None:
        if self._loading:
            return
        cfg = self._config()
        for k, v in asdict(cfg).items():
            if k not in ("source", "connection_id", "llm_model"):
                self.qs.setValue(f"job/{k}", v)

    def _restore(self) -> None:
        self._loading = True
        g = lambda k, d: self.qs.value(f"job/{k}", d)
        b = lambda k, d: str(g(k, d)).lower() in ("true", "1")
        self.count.set_value(min(10, max(1, int(g("clip_count", 8)))))
        self._set_length(float(g("min_len", 20)), float(g("max_len", 60)))
        self.aspect.set_value(str(g("aspect", "9:16")))
        self.cw.setValue(int(g("custom_w", 1080))); self.ch.setValue(int(g("custom_h", 1920)))
        self.captions.set_value(str(g("captions", "Bold")))
        cl = str(g("caption_lang", "auto"))
        if not self.qs.value("caption_auto_default"):        # switch existing installs to Auto once
            cl = "auto"
            self.qs.setValue("caption_auto_default", True)
        self.cap_lang.set_value(cl)
        self.cap_lang.setEnabled(self.captions.value() != "Off")
        self.framing.set_value(str(g("framing", "Auto")))
        self.quality.set_value(str(g("quality", "1080p")))
        self.whisper.set_value(str(g("whisper_model", "auto")))
        self.device.set_value(str(g("device", "auto")))
        self.external.set_value(b("use_external", True))
        self.scenes.set_value(b("use_scenes", True))
        self.cap_quality.set_value(str(g("caption_quality", "best")))
        i = self.language.findData(str(g("language", "")))
        self.language.setCurrentIndex(max(0, i))
        self.out_dir.setText(str(g("output_dir", "")))
        self.custom_row.setVisible(self.aspect.value() == "Custom")
        self.len_row.setVisible(self.length.value() == "custom")
        if str(self.qs.value("advanced_open", "false")).lower() == "true":
            self.adv.setVisible(True); self.adv_btn.setIcon(icon("down"))
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.language.currentIndexChanged.connect(self._options_changed)
        self.out_dir.editingFinished.connect(self._save)
        self.cw.valueChanged.connect(self._options_changed); self.ch.valueChanged.connect(self._options_changed)
        self._loading = False

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self.toast.isVisible():
            self.toast.move((self.width() - self.toast.width()) // 2, self.height() - self.toast.height() - 28)

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker or self.live:
            if QMessageBox.question(self, "Quit", "Something is still running. Stop it and quit?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            if self.worker:
                self.worker.cancel.set()
            if self.live:
                self.live.session.abort.set()
                self.live.session.stop.set()
        if self.studio.uploader.current:
            if QMessageBox.question(self, "Quit", "An upload is in progress. It will continue where it stopped "
                                    "next time you open MyClip. Quit now?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self.studio.shutdown()
        self._save()
        self.qs.setValue("geometry", self.saveGeometry())
        for th in list(getattr(self, "_threads", [])):
            th.quit(); th.wait(4000)
        super().closeEvent(e)
