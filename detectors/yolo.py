"""Jednoprolazni detektor YOLOv4-tiny preko cv2.dnn.

Mreza se izvrsava modulom dnn koji vec dolazi uz OpenCV, pa na plocu nije
potrebno instalirati okruzenje za masinsko ucenje. Cela slika prolazi kroz
mrezu jednom, za razliku od kliznog prozora nad piramidom slike (3.4, 3.5).

Obrada izlaza je vektorizovana preko numpy-a. Mreza za ulaz 416 x 416 daje
2535 kandidata po slici, pa petlja po redovima u Pythonu traje meru
uporedivu sa samim prolazom kroz mrezu -- a to vreme, po granici merenja
utvrdjenoj u base.Detector, ulazi u izmereno vreme zakljucivanja."""

import os

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

    def _detect(self, frame):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)

        # Red je [cx, cy, w, h, objectness, verovatnoce klasa...], a
        # koordinate su relativne u odnosu na dimenzije slike. Verovatnoce
        # klasa su u Darknet izvedbi vec pomnozene merom prisustva objekta,
        # pa se uzimaju neposredno.
        rows = np.concatenate([np.asarray(o).reshape(-1, o.shape[-1])
                               for o in outputs], axis=0)
        scores = rows[:, 5 + self.person_class]

        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []

        rows = rows[keep]
        scores = scores[keep].astype(float)

        cx = rows[:, 0] * width
        cy = rows[:, 1] * height
        bw = rows[:, 2] * width
        bh = rows[:, 3] * height

        boxes = np.stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh],
                         axis=1).astype(int)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold,
        )

        detections = []
        for i in np.asarray(indices).reshape(-1):
            x, y, w, h = boxes[int(i)]
            detections.append(Detection(int(x), int(y), int(w), int(h),
                                        float(scores[int(i)])))
        return detections

    def describe(self):
        return (
            f"YOLOv4-tiny {self.input_size}x{self.input_size} "
            f"conf>={self.conf_threshold} nms={self.nms_threshold}"
        )
