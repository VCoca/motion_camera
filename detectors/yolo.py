"""Jednoprolazni detektor YOLOv4-tiny preko cv2.dnn.

Mreza se izvrsava modulom dnn koji vec dolazi uz OpenCV, pa na plocu nije
potrebno instalirati okruzenje za masinsko ucenje. Cela slika prolazi kroz
mrezu jednom, za razliku od kliznog prozora nad piramidom slike (7.4, 7.5)."""

import os
import time

import cv2
import numpy as np

from .base import Detection, Detector


class YoloDetector(Detector):

    name = "yolo"

    def __init__(self, cfg_path, weights_path, input_size=416,
                 conf_threshold=0.10, nms_threshold=0.45, person_class=0):
        for path in (cfg_path, weights_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Nedostaje model: {path}. Videti models/README.md."
                )

        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.person_class = int(person_class)

        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()

        # Prvi prolaz kroz mrezu je osetno sporiji zbog zauzimanja memorije
        # i pripreme slojeva. Izvodi se ovde da ne bi ulazio u merenje.
        warmup = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect(warmup)

    def detect(self, frame):
        height, width = frame.shape[:2]

        start = time.perf_counter()
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        boxes, scores = [], []
        for output in outputs:
            for row in output:
                # Red je [cx, cy, w, h, objectness, verovatnoce klasa...],
                # a koordinate su relativne u odnosu na dimenzije slike.
                # Verovatnoce klasa su u Darknet izvedbi vec pomnozene
                # merom prisustva objekta, pa se uzimaju neposredno.
                score = float(row[5 + self.person_class])
                if score < self.conf_threshold:
                    continue

                cx, cy, bw, bh = row[0] * width, row[1] * height, \
                    row[2] * width, row[3] * height
                boxes.append([int(cx - bw / 2), int(cy - bh / 2),
                              int(bw), int(bh)])
                scores.append(score)

        detections = []
        if boxes:
            keep = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.nms_threshold
            )
            for i in np.asarray(keep).reshape(-1):
                x, y, w, h = boxes[int(i)]
                detections.append(Detection(x, y, w, h, float(scores[int(i)])))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections, elapsed_ms

    def describe(self):
        return (
            f"YOLOv4-tiny {self.input_size}x{self.input_size} "
            f"conf>={self.conf_threshold} nms={self.nms_threshold}"
        )
