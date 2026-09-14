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
    global _camera

    start = time.perf_counter()

    if _camera is None:
        _camera = _open_camera()

    if _camera is None:
        return None, (time.perf_counter() - start) * 1000.0

    for _ in range(config.CAMERA_FLUSH_FRAMES):
        _camera.grab()

    success, frame = _camera.read()

    if not config.KEEP_CAMERA_OPEN:
        release_camera()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not success or frame is None:
        return None, elapsed_ms

    return frame, elapsed_ms


def draw_boxes(frame, detections):
    """Iscrtava okvire sa merom poverenja uz svaki nalaz."""
    for det in detections:
        cv2.rectangle(frame, (det.x, det.y),
                      (det.x + det.w, det.y + det.h), (0, 255, 0), 2)
        label = f"{det.score:.2f}"
        cv2.putText(frame, label, (det.x, max(det.y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
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
