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

# Broj kadrova koji se odbacuje pre upotrebljivog, da se isprazni
# bafer kamere i ne dobije ustajala slika od pre okidanja.
CAMERA_FLUSH_FRAMES = 4

# Kamera se otvara po okidanju i otpusta posle odluke, cime je
# vreme u kome je aktivna svedeno na kratke odsecke (videti 2.5).
# Za merenja se moze drzati otvorena, jer otvaranje traje osetno.
KEEP_CAMERA_OPEN = False

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
