"""Small, quick animations (300 ms or less). All of them respect Settings -> Reduce motion."""
from __future__ import annotations

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPoint, QPropertyAnimation, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from . import theme

_KEEP: set = set()


def _keep(anim) -> None:
    _KEEP.add(anim)
    anim.finished.connect(lambda: _KEEP.discard(anim))


def on() -> bool:
    return not theme.REDUCE_MOTION


def fade_in(w: QWidget, ms: int = 180, delay: int = 0) -> None:
    if not on():
        w.show()
        return
    eff = QGraphicsOpacityEffect(w)
    eff.setOpacity(0.0)
    w.setGraphicsEffect(eff)
    a = QPropertyAnimation(eff, b"opacity", w)
    a.setDuration(ms)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)

    def done() -> None:
        try:
            w.setGraphicsEffect(None)          # effects cost a lot when scrolling; drop it after
        except RuntimeError:
            pass
    a.finished.connect(done)
    _keep(a)
    if delay:
        QTimer.singleShot(delay, lambda: a.start())
    else:
        a.start()


def slide_up(w: QWidget, end: QPoint, dist: int = 14, ms: int = 200) -> None:
    if not on():
        w.move(end)
        return
    a = QPropertyAnimation(w, b"pos", w)
    a.setDuration(ms)
    a.setStartValue(QPoint(end.x(), end.y() + dist))
    a.setEndValue(end)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)
    _keep(a)
    a.start()
    fade_in(w, ms)


def value(obj, prop: bytes, end, ms: int = 220, start=None) -> None:
    """Animate a property (for example a progress bar's value)."""
    if not on():
        obj.setProperty(prop.decode(), end)
        return
    old = getattr(obj, "_anim_" + prop.decode(), None)
    if old is not None and old.state() == QAbstractAnimation.State.Running:
        old.stop()
    a = QPropertyAnimation(obj, prop, obj)
    a.setDuration(ms)
    a.setStartValue(obj.property(prop.decode()) if start is None else start)
    a.setEndValue(end)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)
    setattr(obj, "_anim_" + prop.decode(), a)
    a.start()


def collapse(w: QWidget, show: bool, ms: int = 220, after=None) -> None:
    """Animate a widget's height open or shut."""
    if not on():
        w.setVisible(show)
        w.setMaximumHeight(16777215)
        if after:
            after()
        return
    if show:
        w.setMaximumHeight(0)
        w.setVisible(True)
        target = max(1, w.sizeHint().height())
        start, end = 0, target
    else:
        start, end = w.height(), 0
    a = QPropertyAnimation(w, b"maximumHeight", w)
    a.setDuration(ms)
    a.setStartValue(start)
    a.setEndValue(end)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)

    def done() -> None:
        if not show:
            w.setVisible(False)
        w.setMaximumHeight(16777215)
        if after:
            after()
    a.finished.connect(done)
    _keep(a)
    a.start()
