"""Visual language: cool charcoal neutrals, one vivid accent, clear status colors.
No gradients, no glows. Every color in the UI comes from this module."""
from __future__ import annotations

# neutrals
SIDEBAR = "#0B0D11"
BG = "#0F1115"
SURFACE = "#171A21"
SURFACE_2 = "#1D212A"
SURFACE_3 = "#232834"
BORDER = "#262B35"
BORDER_HI = "#323846"
TEXT = "#ECEEF2"
TEXT_2 = "#A3A9B6"
TEXT_3 = "#6B7280"
CONTENT_MAX = 1120          # page content width on wide windows
REDUCE_MOTION = False       # set from Settings; animations check it

# status colors (fixed, whatever the accent)
CORAL = "#FF6A3D"
TEAL = "#2DD4BF"
AMBER = "#F5B544"
SKY = "#5AA9FF"
RED = "#F26D6D"

ACCENTS = [("Coral", CORAL), ("Blue", "#4C8DFF"), ("Teal", TEAL), ("Amber", AMBER), ("Pink", "#FF5C8A")]
DEFAULT_ACCENT = CORAL

# accent (changed live by set_accent)
ACCENT = CORAL
ACCENT_HOVER = CORAL
ACCENT_PRESSED = CORAL
ACCENT_TEXT = "#111111"

# old names, still used around the app
SIGNAL = AMBER
DANGER = RED
OK = TEAL


def _rgb(h: str) -> tuple[int, int, int]:
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"not a hex color: {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(h: str, alpha: float) -> str:
    """'#FF6A3D', 0.15 -> 'rgba(255, 106, 61, 38)' (Qt style sheets take alpha 0-255)."""
    r, g, b = _rgb(h)
    a = max(0, min(255, round(alpha * 255)))
    return f"rgba({r}, {g}, {b}, {a})"


def mix(h: str, other: str, t: float) -> str:
    """Blend h toward other by t (0..1)."""
    a, b = _rgb(h), _rgb(other)
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def score_color(score) -> str:
    """Teal for strong clips, amber for decent ones, gray below."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return TEXT_3
    if s >= 75:
        return TEAL
    if s >= 50:
        return AMBER
    return TEXT_3


def set_accent(h: str) -> str:
    """Make h the accent everywhere. Returns the color actually used."""
    global ACCENT, ACCENT_HOVER, ACCENT_PRESSED
    try:
        _rgb(h)
        h = "#" + h.strip().lstrip("#").upper()
    except (ValueError, AttributeError):
        h = DEFAULT_ACCENT
    ACCENT = h
    ACCENT_HOVER = mix(h, "#FFFFFF", 0.14)
    ACCENT_PRESSED = mix(h, "#000000", 0.16)
    return h


def build_qss(accent: str | None = None) -> str:
    if accent:
        set_accent(accent)
    A, AH, AP = ACCENT, ACCENT_HOVER, ACCENT_PRESSED
    tint, tint_hi, tint_lo = rgba(A, 0.15), rgba(A, 0.22), rgba(A, 0.07)
    focus = rgba(A, 0.75)
    return f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {rgba(BORDER, 0.7)}; }}
QWidget#page, QWidget#pagecol {{ background: {BG}; }}
QLabel#brand {{ font-size: 15px; font-weight: 700; padding: 2px 2px; }}
QLabel#h1 {{ font-size: 28px; font-weight: 650; }}
QLabel#h2 {{ font-size: 16px; font-weight: 600; }}
QLabel#h3 {{ font-size: 14px; font-weight: 600; }}
QLabel#lead {{ color: {TEXT_2}; font-size: 14px; }}
QLabel#muted {{ color: {TEXT_2}; }}
QLabel#faint {{ color: {TEXT_3}; font-size: 12px; }}
QLabel#section {{ color: {TEXT_2}; font-size: 11px; font-weight: 700; letter-spacing: 0.9px; }}
QLabel#stat {{ color: {TEXT_3}; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; }}
QLabel#error {{ color: {RED}; }}
QLabel#ok {{ color: {TEAL}; }}
QLabel#info {{ color: {SKY}; }}
QLabel#empty {{ color: {TEXT_3}; font-size: 13px; }}

QPushButton#nav {{ text-align: left; padding: 9px 12px 9px 13px; border: none; border-left: 3px solid transparent;
  border-radius: 8px; color: {TEXT_2}; min-height: 18px; }}
QPushButton#nav:hover {{ background: {SURFACE}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {tint_lo}; color: {TEXT}; border-left: 3px solid {A};
  border-top-left-radius: 0; border-bottom-left-radius: 0; font-weight: 600; }}
QPushButton#nav:focus {{ outline: none; }}

QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 14px; min-height: 18px; }}
QPushButton:hover {{ border-color: {BORDER_HI}; background: {SURFACE_3}; }}
QPushButton:pressed {{ background: {SURFACE}; }}
QPushButton:disabled {{ color: {TEXT_3}; background: {SURFACE}; border-color: {SURFACE_2}; }}
QPushButton:focus {{ outline: none; border-color: {focus}; }}
QPushButton#primary {{ background: {A}; color: {ACCENT_TEXT}; border: 1px solid {A}; font-weight: 600; padding: 9px 22px; }}
QPushButton#primary:hover {{ background: {AH}; border-color: {AH}; }}
QPushButton#primary:pressed {{ background: {AP}; border-color: {AP}; }}
QPushButton#primary:focus {{ border: 1px solid {TEXT}; }}
QPushButton#primary:disabled {{ background: {SURFACE_3}; border-color: {SURFACE_3}; color: {TEXT_3}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_2}; padding: 6px 10px; }}
QPushButton#ghost:hover {{ color: {TEXT}; background: {SURFACE_2}; }}
QPushButton#ghost:pressed {{ background: {tint_lo}; }}
QPushButton#ghost:focus {{ border-color: {focus}; }}
QPushButton#ghost:disabled {{ color: {TEXT_3}; background: transparent; border-color: transparent; }}
QPushButton#danger {{ background: transparent; color: {RED}; border: 1px solid {BORDER}; }}
QPushButton#danger:hover {{ background: {rgba(RED, 0.12)}; border-color: {RED}; }}
QPushButton#icon {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 6px; }}
QPushButton#icon:hover {{ background: {SURFACE_3}; }}
QPushButton#overlay {{ background: rgba(10, 11, 14, 170); border: none; border-radius: 8px; padding: 6px; }}
QPushButton#overlay:hover {{ background: rgba(10, 11, 14, 220); }}

QPushButton#chip {{ background: transparent; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 13px; color: {TEXT_2}; }}
QPushButton#chip:hover {{ color: {TEXT}; border-color: {BORDER_HI}; background: {tint_lo}; }}
QPushButton#chip:checked {{ color: {A}; background: {tint}; border-color: {rgba(A, 0.85)}; font-weight: 600; }}
QPushButton#chip:checked:hover {{ background: {tint_hi}; }}
QPushButton#chip:focus {{ border-color: {focus}; }}
QPushButton#chip:disabled {{ color: {TEXT_3}; border-color: {SURFACE_2}; background: transparent; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QDateTimeEdit {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 11px; min-height: 18px;
  selection-background-color: {rgba(A, 0.35)};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTextEdit:hover, QPlainTextEdit:hover,
QDateTimeEdit:hover {{ border-color: {BORDER_HI}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus,
QDateTimeEdit:focus {{ border-color: {A}; }}
QLineEdit#url {{ font-size: 14px; padding: 11px 14px; border-radius: 10px; }}
QLineEdit#search {{ padding-left: 12px; border-radius: 10px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER_HI}; selection-background-color: {tint};
  selection-color: {TEXT}; outline: none; padding: 4px; border-radius: 8px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 8px; color: {TEXT_2}; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER_HI}; border-radius: 5px; background: {SURFACE}; }}
QCheckBox::indicator:hover {{ border-color: {A}; }}
QCheckBox::indicator:checked {{ background: {A}; border-color: {A}; }}
QCheckBox#cardcheck::indicator {{ width: 20px; height: 20px; border-radius: 6px; background: rgba(10, 11, 14, 170);
  border: 1px solid rgba(255, 255, 255, 90); }}
QCheckBox#cardcheck::indicator:checked {{ background: {A}; border-color: {A}; }}

QProgressBar {{ background: {SURFACE_3}; border: none; border-radius: 3px; height: 6px; max-height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {A}; border-radius: 3px; }}
QProgressBar#voice::chunk {{ background: {CORAL}; }}
QProgressBar#chat::chunk {{ background: {SKY}; }}
QProgressBar#done::chunk {{ background: {TEAL}; }}

QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_3}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {A}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {TEXT}; }}
QSlider::handle:horizontal:hover {{ background: {AH}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_3}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_HI}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QScrollBar:horizontal {{ height: 0; }}

QFrame#card {{ background: {SURFACE}; border: 1px solid transparent; border-radius: 12px; }}
QFrame#card:hover {{ border-color: {BORDER_HI}; }}
QFrame#card[selected="true"] {{ border-color: {A}; background: {mix(SURFACE, A, 0.06)}; }}
QFrame#line {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#panel {{ background: {SURFACE}; border: 1px solid {rgba(BORDER, 0.8)}; border-radius: 14px; }}
QFrame#soft {{ background: {SURFACE}; border: none; border-radius: 14px; }}
QFrame#selbar {{ background: {SURFACE_2}; border: 1px solid {BORDER_HI}; border-radius: 12px; }}

QTabBar::tab {{ background: transparent; color: {TEXT_2}; padding: 8px 14px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {A}; font-weight: 600; }}
QTabWidget::pane {{ border: none; }}
QTableWidget, QListWidget, QTreeWidget {{ background: transparent; border: none; gridline-color: {BORDER}; outline: none; }}
QTableWidget::item, QListWidget::item {{ padding: 4px 6px; border-radius: 6px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {tint}; color: {TEXT}; }}
QHeaderView::section {{ background: transparent; color: {TEXT_3}; border: none; border-bottom: 1px solid {BORDER};
  padding: 6px; font-size: 11px; font-weight: 600; }}

QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 6px 8px; border-radius: 6px; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER_HI}; padding: 5px; border-radius: 10px; }}
QMenu::item {{ padding: 7px 16px; border-radius: 6px; }}
QMenu::item:selected {{ background: {tint}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QDialog {{ background: {SURFACE}; }}
"""


set_accent(DEFAULT_ACCENT)
QSS = build_qss()        # kept for older imports
