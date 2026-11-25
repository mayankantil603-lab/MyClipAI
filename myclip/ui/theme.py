"""Visual language: cool charcoal neutrals, one vivid accent, clear status colors.
No gradients, no glows. Every color in the UI comes from this module."""
from __future__ import annotations

# neutrals
BG = "#0F1115"
SURFACE = "#171A21"
SURFACE_2 = "#1D212A"
SURFACE_3 = "#232834"
BORDER = "#262B35"
BORDER_HI = "#323846"
TEXT = "#ECEEF2"
TEXT_2 = "#A3A9B6"
TEXT_3 = "#6B7280"

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
    return f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
QWidget#page {{ background: {BG}; }}
QLabel#brand {{ font-size: 15px; font-weight: 700; padding: 2px 4px; }}
QLabel#h1 {{ font-size: 26px; font-weight: 700; }}
QLabel#h2 {{ font-size: 16px; font-weight: 600; }}
QLabel#muted {{ color: {TEXT_2}; }}
QLabel#faint {{ color: {TEXT_3}; font-size: 12px; }}
QLabel#section {{ color: {TEXT_2}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; }}
QLabel#stat {{ color: {TEXT_3}; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; }}
QLabel#error {{ color: {RED}; }}
QLabel#ok {{ color: {TEAL}; }}
QLabel#info {{ color: {SKY}; }}

QPushButton#nav {{ text-align: left; padding: 8px 10px 8px 12px; border: none; border-left: 3px solid transparent;
  border-radius: 6px; color: {TEXT_2}; }}
QPushButton#nav:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {tint}; color: {TEXT}; border-left: 3px solid {A};
  border-top-left-radius: 0; border-bottom-left-radius: 0; font-weight: 600; }}

QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {BORDER_HI}; background: {SURFACE_3}; }}
QPushButton:pressed {{ background: {SURFACE}; }}
QPushButton:disabled {{ color: {TEXT_3}; background: {SURFACE}; }}
QPushButton:focus {{ outline: none; border-color: {rgba(A, 0.6)}; }}
QPushButton#primary {{ background: {A}; color: {ACCENT_TEXT}; border: 1px solid {A}; font-weight: 700; padding: 8px 20px; }}
QPushButton#primary:hover {{ background: {AH}; border-color: {AH}; }}
QPushButton#primary:pressed {{ background: {AP}; border-color: {AP}; }}
QPushButton#primary:disabled {{ background: {SURFACE_3}; border-color: {SURFACE_3}; color: {TEXT_3}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_2}; padding: 5px 8px; }}
QPushButton#ghost:hover {{ color: {TEXT}; background: {SURFACE_2}; }}
QPushButton#ghost:pressed {{ background: {tint_lo}; }}
QPushButton#danger {{ background: transparent; color: {RED}; border: 1px solid {BORDER}; }}
QPushButton#danger:hover {{ background: {rgba(RED, 0.12)}; border-color: {RED}; }}

QPushButton#chip {{ background: transparent; border: 1px solid {BORDER}; border-radius: 7px; padding: 5px 12px; color: {TEXT_2}; }}
QPushButton#chip:hover {{ color: {TEXT}; border-color: {BORDER_HI}; background: {tint_lo}; }}
QPushButton#chip:checked {{ color: {A}; background: {tint}; border-color: {A}; font-weight: 600; }}
QPushButton#chip:checked:hover {{ background: {tint_hi}; }}
QPushButton#chip:disabled {{ color: {TEXT_3}; border-color: {BORDER}; background: transparent; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QDateTimeEdit {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 9px;
  selection-background-color: {rgba(A, 0.35)};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTextEdit:hover, QDateTimeEdit:hover {{ border-color: {BORDER_HI}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QDateTimeEdit:focus {{ border-color: {A}; }}
QLineEdit#url {{ font-size: 14px; padding: 10px 13px; background: {SURFACE}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER_HI}; selection-background-color: {tint};
  selection-color: {TEXT}; outline: none; padding: 4px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 8px; color: {TEXT_2}; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER_HI}; border-radius: 4px; background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: {A}; border-color: {A}; }}

QProgressBar {{ background: {SURFACE_3}; border: none; border-radius: 3px; height: 6px; max-height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {A}; border-radius: 3px; }}
QProgressBar#voice::chunk {{ background: {CORAL}; }}
QProgressBar#chat::chunk {{ background: {SKY}; }}
QProgressBar#done::chunk {{ background: {TEAL}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_3}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_HI}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QScrollBar:horizontal {{ height: 0; }}

QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card:hover {{ border-color: {A}; }}
QFrame#line {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}

QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 5px 7px; border-radius: 4px; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER_HI}; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {tint}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QDialog {{ background: {SURFACE}; }}
QListWidget {{ background: transparent; border: none; }}
"""


set_accent(DEFAULT_ACCENT)
QSS = build_qss()        # kept for older imports
