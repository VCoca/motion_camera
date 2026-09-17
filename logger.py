"""Evidencija dogadjaja u CSV obliku.

Jedan red po okidanju senzora, sa svim vremenima potrebnim za merenja u
radu. Veb strana cita istu datoteku, pa stanje sistema zaista zivi u
sacuvanim podacima, a ne u protokolu (8.3 istrazivackog rada)."""

import csv
import os
import subprocess
from datetime import datetime

import config

FIELDS = [
    "ts",             # trenutak okidanja, citljiv zapis
    "detector",       # hog | yolo | -
    "width",          # dimenzije obradjenog kadra
    "height",
    "t_capture_ms",   # t1 - t0, otvaranje kamere i akvizicija
    "t_infer_ms",     # t2 - t1, zakljucivanje detektora
    "t_total_ms",     # t3 - t0, odziv sistema
    "n_boxes",        # broj nalaza iznad praga
    "max_score",      # najveca mera poverenja
    "decision",       # person | no_person | camera_error | warmup
    "image",          # ime sacuvane datoteke ili prazno
    "cpu_temp_c",
    "throttled",      # stanje ogranicavanja takta, vcgencmd get_throttled
]

# Zaglavlje datoteke sa prikupljenim kadrovima. Scenario se zadaje pri
# pokretanju (--scenario), jer se posle snimanja ne moze rekonstruisati.
FRAME_FIELDS = [
    "frame_id",       # ime sacuvane datoteke
    "ts",
    "scenario",       # npr. 4m-popreko-dnevno
    "width",
    "height",
    "decision",       # sta je detektor rekao, radi obavestenja pri oznacavanju
]


def _ensure_file():
    """Pravi evidenciju ako je nema. Ako postoji, a zaglavlje joj se ne
    poklapa sa FIELDS (stara evidencija posle dodavanja kolone), odlaze se u
    stranu: dopisivanje redova sa novim brojem kolona pod starim zaglavljem
    daje datoteku koju nijedan alat ne cita ispravno."""
    os.makedirs(config.LOG_FOLDER, exist_ok=True)

    if os.path.exists(config.EVENT_CSV):
        with open(config.EVENT_CSV, newline="") as handle:
            header = next(csv.reader(handle), [])
        if header == FIELDS:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{config.EVENT_CSV}.{stamp}.old"
        os.replace(config.EVENT_CSV, backup)
        print(f"evidencija je imala staro zaglavlje, sacuvana kao {backup}",
              flush=True)

    with open(config.EVENT_CSV, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()


def cpu_temperature():
    """Temperatura sistema na cipu, u stepenima Celzijusa."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as handle:
            return round(int(handle.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return ""


def throttled_state():
    """Stanje ogranicavanja takta; koristi se u izvestaju o merenjima."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def log_event(detector="-", width="", height="", t_capture_ms=None,
              t_infer_ms=None, t_total_ms=None, n_boxes=0, max_score=None,
              decision="", image=""):
    """Upisuje red u evidenciju. Poziva se posle merenja vremena, pa
    citanje temperature i stanja takta ne ulazi u vreme odziva."""
    _ensure_file()

    def ms(value):
        return "" if value is None else f"{value:.1f}"

    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detector": detector,
        "width": width,
        "height": height,
        "t_capture_ms": ms(t_capture_ms),
        "t_infer_ms": ms(t_infer_ms),
        "t_total_ms": ms(t_total_ms),
        "n_boxes": n_boxes,
        "max_score": "" if max_score is None else f"{max_score:.4f}",
        "decision": decision,
        "image": image,
        "cpu_temp_c": cpu_temperature(),
        "throttled": throttled_state(),
    }

    with open(config.EVENT_CSV, "a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)

    return row


def log_frame(frame_id, scenario="", width="", height="", decision=""):
    """Upisuje red u frames.csv za sliku prikupljenu u rezimu --save-mode all."""
    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    exists = os.path.exists(config.FRAMES_CSV)

    with open(config.FRAMES_CSV, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "frame_id": frame_id,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "width": width,
            "height": height,
            "decision": decision,
        })


def read_events(limit=None):
    """Vraca dogadjaje, najnoviji prvi."""
    if not os.path.exists(config.EVENT_CSV):
        return []

    with open(config.EVENT_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows.reverse()
    return rows[:limit] if limit else rows


def write_status(state, detector="-"):
    """Stanje sa vremenom upisa, da veb strana moze da proceni da li je
    sistem ziv."""
    with open(config.STATUS_FILE, "w") as handle:
        handle.write(f"{state}\n{datetime.now().isoformat(timespec='seconds')}\n"
                     f"{detector}\n")


def read_status():
    """Vraca (stanje, vreme_upisa_iso, detektor)."""
    try:
        with open(config.STATUS_FILE) as handle:
            parts = handle.read().strip().split("\n")
    except OSError:
        return ("Nepoznato", "", "-")

    parts += [""] * (3 - len(parts))
    return (parts[0] or "Nepoznato", parts[1], parts[2] or "-")
