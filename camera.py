"""Akvizicija slike sa USB kamere preko V4L2.

Kamera se otvara po okidanju senzora i otpusta posle odluke, cime je vreme
u kome je aktivna svedeno na kratke odsecke (2.5 istrazivackog rada). Pre
upotrebljivog kadra prazni se bafer, jer UVC kamera inace vrati zapamcen
kadar iz vremena pre okidanja."""

import os
import time
from datetime import datetime

import cv2

import config

_camera = None
_last_used = 0.0


def _open_camera():
    cam = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_V4L2)
    if not cam.isOpened():
        cam.release()
        return None

    # MJPEG jer nekompresovan format na USB 2.0 ogranicava broj kadrova.
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    # Ne podrzavaju ga svi upravljacki programi, pa je praznjenje bafera
    # ispod pravo resenje, a ovo samo pomoc kada radi.
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cam


def capture_frame():
    """Vraca (kadar ili None, trajanje_akvizicije_ms).

    Trajanje obuhvata i otvaranje kamere, jer je i ono deo vremena odziva
    sistema kada kamera nije trajno otvorena."""
    global _camera, _last_used

    start = time.perf_counter()

    was_open = _camera is not None
    if not was_open:
        _camera = _open_camera()

    if _camera is None:
        return None, (time.perf_counter() - start) * 1000.0

    flush = (config.CAMERA_FLUSH_FRAMES_WARM if was_open
             else config.CAMERA_FLUSH_FRAMES)
    for _ in range(flush):
        _camera.grab()

    success, frame = _camera.read()
    _last_used = time.monotonic()

    if not config.KEEP_CAMERA_OPEN:
        release_camera()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not success or frame is None:
        return None, elapsed_ms

    return frame, elapsed_ms


def draw_boxes(frame, detections, color=(0, 255, 0), prefix=""):
    """Iscrtava okvire sa merom poverenja uz svaki nalaz.

    Podrazumevane vrednosti daju isti prikaz kao u radu sistema. Boja i
    natpis se menjaju samo kad se na istu sliku iscrtavaju nalazi dva
    postupka, radi poredjenja (vidi eval/draw.py).
    """
    for det in detections:
        cv2.rectangle(frame, (det.x, det.y),
                      (det.x + det.w, det.y + det.h), color, 2)
        label = f"{prefix}{det.score:.2f}" if prefix else f"{det.score:.2f}"
        cv2.putText(frame, label, (det.x, max(det.y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)


def save_frame(frame, prefix="motion"):
    """Upisuje kadar i vraca ime datoteke (bez putanje)."""
    os.makedirs(config.PHOTO_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(config.PHOTO_FOLDER, filename)

    cv2.imwrite(path, frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
    return filename


def release_camera():
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


def release_if_idle():
    """Otpusta kameru posle zadatog vremena bez okidanja. Poziva se iz
    glavne petlje dok ona ceka na senzor, cime kamera u praznoj prostoriji
    ostaje zatvorena, a tokom prolaska osobe se otvaranje placa samo
    jednom. Vraca True ako je kamera otpustena."""
    if _camera is None or not config.CAMERA_IDLE_RELEASE_S:
        return False

    if time.monotonic() - _last_used < config.CAMERA_IDLE_RELEASE_S:
        return False

    release_camera()
    return True


def is_open():
    return _camera is not None
