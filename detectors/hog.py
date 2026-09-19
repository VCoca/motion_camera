"""HOG deskriptor sa linearnim SVM klasifikatorom.

Koristi se detektor pesaka koji dolazi uz OpenCV, obucen nad prozorom
64 x 128 piksela (Dalal i Triggs, 2005). Posledica te velicine prozora:
osoba niza od 128 piksela u ulaznoj slici ne moze biti otkrivena."""

import cv2
import numpy as np

from .base import Detection, Detector


class HogDetector(Detector):

    name = "hog"

    # Najmanja visina osobe u pikselima koju prozor 64 x 128 uopste moze
    # da obuhvati. Koristi se u izvestaju o dometu kamere.
    MIN_PERSON_HEIGHT_PX = 128

    def __init__(self, win_stride=(8, 8), padding=(8, 8), scale=1.05,
                 box_shrink=(1.0, 1.0)):
        self.win_stride = tuple(win_stride)
        self.padding = tuple(padding)
        self.scale = float(scale)
        self.box_shrink = (float(box_shrink[0]), float(box_shrink[1]))

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            # Prag se ne postavlja ovde nego nad skorovima, da bi se
            # kriva preciznosti i odziva mogla nacrtati iz jednog prolaza.
            hitThreshold=0.0,
        )

        scores = (np.asarray(weights, dtype=float).reshape(-1)
                  if len(weights) else [])

        height, width = frame.shape[:2]
        fw, fh = self.box_shrink

        detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            score = float(scores[i]) if i < len(scores) else 0.0

            # Prozor se svodi na okvir oko same osobe (videti config.py), pa
            # se odseca na granice kadra.
            cx, cy = x + w / 2.0, y + h / 2.0
            nw, nh = w * fw, h * fh
            x0 = max(0, int(round(cx - nw / 2.0)))
            y0 = max(0, int(round(cy - nh / 2.0)))
            x1 = min(width, int(round(cx + nw / 2.0)))
            y1 = min(height, int(round(cy + nh / 2.0)))
            if x1 <= x0 or y1 <= y0:
                continue

            detections.append(Detection(x0, y0, x1 - x0, y1 - y0, score))
        return detections

    def describe(self):
        return (
            f"HOG+SVM winStride={self.win_stride} "
            f"padding={self.padding} scale={self.scale} "
            f"svodjenje okvira={self.box_shrink}"
        )
