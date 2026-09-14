"""HOG dekriptor sa linearnim SVM klasifikatorom.

Koristi se detektor pesaka koji dolazi uz OpenCV, obucen nad prozorom
64 x 128 piksela (Dalal i Triggs, 2005). Posledica te velicine prozora:
osoba niza od 128 piksela u ulaznoj slici ne moze biti otkrivena."""

import time

import cv2
import numpy as np

from .base import Detection, Detector


class HogDetector(Detector):

    name = "hog"

    # Najmanja visina osobe u pikselima koju prozor 64 x 128 uopste moze
    # da obuhvati. Koristi se u izvestaju o dometu kamere.
    MIN_PERSON_HEIGHT_PX = 128

    def __init__(self, win_stride=(8, 8), padding=(8, 8), scale=1.05):
        self.win_stride = tuple(win_stride)
        self.padding = tuple(padding)
        self.scale = float(scale)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        start = time.perf_counter()
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            # Prag se ne postavlja ovde nego nad skorovima, da bi se
            # kriva preciznosti i odziva mogla nacrtati iz jednog prolaza.
            hitThreshold=0.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        scores = np.asarray(weights, dtype=float).reshape(-1) if len(weights) else []

        detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            score = float(scores[i]) if i < len(scores) else 0.0
            detections.append(Detection(int(x), int(y), int(w), int(h), score))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections, elapsed_ms

    def describe(self):
        return (
            f"HOG+SVM winStride={self.win_stride} "
            f"padding={self.padding} scale={self.scale}"
        )
