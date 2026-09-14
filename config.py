"""Podešavanja sistema. Sve putanje su izvedene iz položaja ovog fajla,
da bi sistem radio i kada ga pokrene systemd iz drugog radnog direktorijuma."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- GPIO ---
GPIO_PIN = 17
LED_PIN = 18

# HC-SR501 posle ukljucenja napajanja daje do tri lazna impulsa
# u periodu inicijalizacije od oko jednog minuta, pa se detekcije
# u tom vremenu odbacuju (videti 4.6.2 istrazivackog rada).
WARMUP_S = 60

# --- Kamera ---
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Broj kadrova koji se odbacuje pre upotrebljivog. Dve razlicite svrhe,
# pa i dve vrednosti (obe potvrdjene merenjem, videti README):
#   - posle otvaranja kamere automatika ekspozicije se smiruje tek oko
#     cetvrtog kadra, pa je prvi kadar i do 50 % svetliji od smirenog;
#   - kod vec otvorene kamere odbacuje se zapamcen kadar iz bafera, da
#     se ne dobije slika od pre okidanja.
CAMERA_FLUSH_FRAMES = 4       # tek otvorena kamera
CAMERA_FLUSH_FRAMES_WARM = 2  # kamera vec otvorena i smirena

# Drzanje kamere otvorene izmedju okidanja skracuje vreme akvizicije sa
# oko 1200 ms na oko 170 ms, jer otvaranje uredjaja samo po sebi traje
# preko pola sekunde. Da argument o privatnosti iz odeljka 2.5 ostane na
# snazi, kamera se otpusta posle CAMERA_IDLE_RELEASE_S sekundi bez
# okidanja: tokom prolaska osobe placa se otvaranje jednom, a u praznoj
# prostoriji kamera je zatvorena.
KEEP_CAMERA_OPEN = True
CAMERA_IDLE_RELEASE_S = 30.0  # 0 iskljucuje otpustanje

# --- Detektor ---
DETECTOR = "hog"          # "hog" ili "yolo"
SCORE_THRESHOLD = 0.0     # radna tacka; 0.0 znaci "prihvati sve nalaze"

# HOG
HOG_WIN_STRIDE = (8, 8)
HOG_PADDING = (8, 8)
HOG_SCALE = 1.05

# YOLO (cv2.dnn, Darknet)
YOLO_CFG = os.path.join(BASE_DIR, "models", "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "models", "yolov4-tiny.weights")
YOLO_INPUT_SIZE = 416
YOLO_CONF_THRESHOLD = 0.10   # nisko namerno: radna tacka se bira nad skorovima
YOLO_NMS_THRESHOLD = 0.45
YOLO_PERSON_CLASS = 0        # indeks klase "person" u COCO skupu

# --- Cuvanje ---
# "person" -> cuva se samo kada je osoba nadjena (radni rezim)
# "all"    -> cuva se svaki kadar posle okidanja (rezim prikupljanja skupa)
SAVE_MODE = "person"

PHOTO_FOLDER = os.path.join(BASE_DIR, "photos")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
EVENT_CSV = os.path.join(LOG_FOLDER, "events.csv")
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")

JPEG_QUALITY = 90

# Najkrace vreme izmedju dve obrade, u sekundama.
COOLDOWN = 1.0
