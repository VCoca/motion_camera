"""Izbor detektora po imenu, da bi ostatak sistema bio nezavisan od toga
koji je postupak u pogonu."""

import config

from .base import Detection, Detector, filter_by_score

AVAILABLE = ("hog", "yolo")


def create_detector(name=None):
    name = (name or config.DETECTOR).lower()

    if name == "hog":
        from .hog import HogDetector
        return HogDetector(
            win_stride=config.HOG_WIN_STRIDE,
            padding=config.HOG_PADDING,
            scale=config.HOG_SCALE,
        )

    if name == "yolo":
        from .yolo import YoloDetector
        return YoloDetector(
            cfg_path=config.YOLO_CFG,
            weights_path=config.YOLO_WEIGHTS,
            input_size=config.YOLO_INPUT_SIZE,
            conf_threshold=config.YOLO_CONF_THRESHOLD,
            nms_threshold=config.YOLO_NMS_THRESHOLD,
            person_class=config.YOLO_PERSON_CLASS,
        )

    raise ValueError(f"Nepoznat detektor: {name}. Dostupni: {AVAILABLE}")


__all__ = ["Detection", "Detector", "create_detector", "filter_by_score",
           "AVAILABLE"]
