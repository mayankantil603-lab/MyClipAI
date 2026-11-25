"""A few line icons drawn with QPainter, so there are no image assets to lose."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from . import theme


def _pix(size: int, color: str, draw) -> QPixmap:
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.scale(size / 24, size / 24)
    draw(p)
    p.end()
    return pm


def _create(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 3, 3)
    p.drawLine(QPointF(12, 8.5), QPointF(12, 15.5))
    p.drawLine(QPointF(8.5, 12), QPointF(15.5, 12))


def _library(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 5, 7, 14), 2, 2)
    p.drawRoundedRect(QRectF(13, 5, 7, 6), 2, 2)
    p.drawRoundedRect(QRectF(13, 13, 7, 6), 2, 2)


def _settings(p: QPainter) -> None:
    for y, x in ((7, 9), (12, 15), (17, 7)):
        p.drawLine(QPointF(4, y), QPointF(20, y))
        p.setBrush(QColor(theme.SURFACE))
        p.drawEllipse(QPointF(x, y), 2.2, 2.2)
        p.setBrush(Qt.BrushStyle.NoBrush)


def _upload(p: QPainter) -> None:
    p.drawLine(QPointF(12, 15), QPointF(12, 4.5))
    p.drawLine(QPointF(8, 8.5), QPointF(12, 4.5))
    p.drawLine(QPointF(16, 8.5), QPointF(12, 4.5))
    path = QPainterPath(QPointF(4.5, 14))
    path.lineTo(4.5, 19.5)
    path.lineTo(19.5, 19.5)
    path.lineTo(19.5, 14)
    p.drawPath(path)


def _play(p: QPainter) -> None:
    path = QPainterPath(QPointF(8, 5.5))
    path.lineTo(18.5, 12)
    path.lineTo(8, 18.5)
    path.closeSubpath()
    p.drawPath(path)


def _folder(p: QPainter) -> None:
    path = QPainterPath(QPointF(4, 7))
    path.lineTo(4, 18.5)
    path.lineTo(20, 18.5)
    path.lineTo(20, 8.5)
    path.lineTo(12, 8.5)
    path.lineTo(10, 6)
    path.lineTo(4, 6)
    path.closeSubpath()
    p.drawPath(path)


def _copy(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(8, 8, 11, 11), 2, 2)
    p.drawLine(QPointF(5, 15), QPointF(5, 6))
    p.drawLine(QPointF(5, 5), QPointF(15, 5))


def _chevron(p: QPainter) -> None:
    p.drawLine(QPointF(9, 6), QPointF(15, 12))
    p.drawLine(QPointF(15, 12), QPointF(9, 18))


def _down(p: QPainter) -> None:
    p.drawLine(QPointF(6, 9), QPointF(12, 15))
    p.drawLine(QPointF(12, 15), QPointF(18, 9))


def _plus(p: QPainter) -> None:
    p.drawLine(QPointF(12, 6), QPointF(12, 18))
    p.drawLine(QPointF(6, 12), QPointF(18, 12))


def _live(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 2.2, 2.2)
    p.drawArc(QRectF(7, 7, 10, 10), 40 * 16, 100 * 16)
    p.drawArc(QRectF(7, 7, 10, 10), 220 * 16, 100 * 16)
    p.drawArc(QRectF(3.5, 3.5, 17, 17), 40 * 16, 100 * 16)
    p.drawArc(QRectF(3.5, 3.5, 17, 17), 220 * 16, 100 * 16)


def _studio(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.5, 6, 17, 12), 3, 3)
    path = QPainterPath(QPointF(10.5, 9.5))
    path.lineTo(14.5, 12)
    path.lineTo(10.5, 14.5)
    path.closeSubpath()
    p.drawPath(path)


def _pause(p: QPainter) -> None:
    p.drawLine(QPointF(9, 6), QPointF(9, 18))
    p.drawLine(QPointF(15, 6), QPointF(15, 18))


def _prev(p: QPainter) -> None:
    p.drawLine(QPointF(7, 6), QPointF(7, 18))
    path = QPainterPath(QPointF(18, 6))
    path.lineTo(10, 12)
    path.lineTo(18, 18)
    path.closeSubpath()
    p.drawPath(path)


def _next(p: QPainter) -> None:
    p.drawLine(QPointF(17, 6), QPointF(17, 18))
    path = QPainterPath(QPointF(6, 6))
    path.lineTo(14, 12)
    path.lineTo(6, 18)
    path.closeSubpath()
    p.drawPath(path)


def _scissors(p: QPainter) -> None:
    p.drawEllipse(QPointF(7, 17), 2.6, 2.6)
    p.drawEllipse(QPointF(17, 17), 2.6, 2.6)
    p.drawLine(QPointF(8.8, 15.2), QPointF(17, 5))
    p.drawLine(QPointF(15.2, 15.2), QPointF(7, 5))


def _edit(p: QPainter) -> None:
    path = QPainterPath(QPointF(5, 19))
    path.lineTo(5.6, 15.4)
    path.lineTo(15.5, 5.5)
    path.lineTo(18.5, 8.5)
    path.lineTo(8.6, 18.4)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(13.5, 7.5), QPointF(16.5, 10.5))


def _trash(p: QPainter) -> None:
    p.drawLine(QPointF(5, 7), QPointF(19, 7))
    p.drawLine(QPointF(10, 7), QPointF(10.5, 4.5))
    p.drawLine(QPointF(10.5, 4.5), QPointF(13.5, 4.5))
    p.drawLine(QPointF(13.5, 4.5), QPointF(14, 7))
    path = QPainterPath(QPointF(6.5, 7))
    path.lineTo(7.5, 19.5)
    path.lineTo(16.5, 19.5)
    path.lineTo(17.5, 7)
    p.drawPath(path)


def _check(p: QPainter) -> None:
    p.drawLine(QPointF(5.5, 12.5), QPointF(10, 17))
    p.drawLine(QPointF(10, 17), QPointF(18.5, 7.5))


def _close(p: QPainter) -> None:
    p.drawLine(QPointF(7, 7), QPointF(17, 17))
    p.drawLine(QPointF(17, 7), QPointF(7, 17))


def _volume(p: QPainter) -> None:
    path = QPainterPath(QPointF(4.5, 9.5))
    path.lineTo(8, 9.5)
    path.lineTo(12.5, 5.5)
    path.lineTo(12.5, 18.5)
    path.lineTo(8, 14.5)
    path.lineTo(4.5, 14.5)
    path.closeSubpath()
    p.drawPath(path)
    p.drawArc(QRectF(11, 8, 7, 8), -60 * 16, 120 * 16)


def _film(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 5, 16, 14), 2.5, 2.5)
    p.drawLine(QPointF(8, 5), QPointF(8, 19))
    p.drawLine(QPointF(16, 5), QPointF(16, 19))
    p.drawLine(QPointF(4, 12), QPointF(8, 12))
    p.drawLine(QPointF(16, 12), QPointF(20, 12))


def _search(p: QPainter) -> None:
    p.drawEllipse(QPointF(10.5, 10.5), 5.5, 5.5)
    p.drawLine(QPointF(14.5, 14.5), QPointF(19, 19))


def _person(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 9), 3.6, 3.6)
    path = QPainterPath(QPointF(5, 19.5))
    path.cubicTo(6, 14.5, 18, 14.5, 19, 19.5)
    p.drawPath(path)


def _cpu(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(7, 7, 10, 10), 2, 2)
    for k in (9.5, 14.5):
        p.drawLine(QPointF(k, 4), QPointF(k, 7))
        p.drawLine(QPointF(k, 17), QPointF(k, 20))
        p.drawLine(QPointF(4, k), QPointF(7, k))
        p.drawLine(QPointF(17, k), QPointF(20, k))


_DRAW = {"studio": _studio, "live": _live, "create": _create, "library": _library, "settings": _settings, "upload": _upload,
         "play": _play, "folder": _folder, "copy": _copy, "chevron": _chevron, "down": _down, "plus": _plus,
         "pause": _pause, "prev": _prev, "next": _next, "scissors": _scissors, "edit": _edit, "trash": _trash,
         "check": _check, "close": _close, "volume": _volume, "film": _film, "search": _search,
         "person": _person, "cpu": _cpu}


def icon(name: str, color: str = "", size: int = 18, on_color: str = "") -> QIcon:
    """Line icon. `on_color` is used while a checkable button is checked (e.g. the active page)."""
    color = color or theme.TEXT_2
    ic = QIcon()
    ic.addPixmap(_pix(size, color, _DRAW[name]), QIcon.Mode.Normal)
    ic.addPixmap(_pix(size, theme.TEXT, _DRAW[name]), QIcon.Mode.Active)
    ic.addPixmap(_pix(size, theme.TEXT, _DRAW[name]), QIcon.Mode.Selected)
    if on_color:
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
            ic.addPixmap(_pix(size, on_color, _DRAW[name]), mode, QIcon.State.On)
    return ic


def pixmap(name: str, color: str, size: int) -> QPixmap:
    return _pix(size, color, _DRAW[name])


def app_icon() -> QIcon:
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(s, s)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)
        p.setBrush(QColor(theme.ACCENT_TEXT))
        r = s * 0.17
        p.drawRoundedRect(QRectF(s * 0.30, s * 0.20, s * 0.40, s * 0.60), r * 0.6, r * 0.6)
        p.setBrush(QColor(theme.ACCENT))
        path = QPainterPath(QPointF(s * 0.44, s * 0.40))
        path.lineTo(s * 0.60, s * 0.50)
        path.lineTo(s * 0.44, s * 0.60)
        path.closeSubpath()
        p.drawPath(path)
        p.end()
        ic.addPixmap(pm)
    return ic
