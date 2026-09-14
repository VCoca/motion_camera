# Model za jednoprolazni detektor

Koristi se **YOLOv4-tiny** u Darknet obliku, koji se izvršava modulom
`cv2.dnn`. Zahvaljujući tome na ploču nije potrebno instalirati okruženje za
mašinsko učenje (`torch` sa zavisnostima prelazi 1 GB, a slobodne radne
memorije ima oko 1,3 GB).

Datoteke u ovoj mapi:

| Datoteka | Veličina | Izvor |
|----------|---------|-------|
| `yolov4-tiny.cfg` | 3 KB | https://github.com/AlexeyAB/darknet — `cfg/yolov4-tiny.cfg` |
| `yolov4-tiny.weights` | 23 MB | https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights |
| `coco.names` | 625 B | https://github.com/AlexeyAB/darknet — `data/coco.names` |

Model je obučen nad skupom COCO, sa 80 klasa; sistem koristi samo klasu
`person`, koja je u tom skupu pod indeksom 0 (`YOLO_PERSON_CLASS` u
`config.py`).

Ponovno preuzimanje:

```bash
curl -fL -o models/yolov4-tiny.cfg \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
curl -fL -o models/coco.names \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
curl -fL -o models/yolov4-tiny.weights \
  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

Težine se ne drže u repozitorijumu (videti `.gitignore`).
