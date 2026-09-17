"""Pokretanje i zaustavljanje glavne petlje sa veb strane.

Nadzor prostorije i veb prikaz su dva odvojena procesa: jedan drzi GPIO
linije i kameru, drugi usluzuje zahteve pregledaca. Veb proces zato ne
poziva glavnu petlju nego je pokrece kao poseban proces i pamti njegov
broj u datoteci, tako da stanje prezivi i ponovno pokretanje servera.

Isti razlog kao i kod evidencije: posto HTTP nema stanje, sve od cega
odgovor zavisi mora se pri svakom zahtevu procitati sa diska (8.3)."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import config

PID_FILE = os.path.join(config.LOG_FOLDER, "main.pid")
SYSTEM_LOG = os.path.join(config.LOG_FOLDER, "system.log")

DETECTORS = ("hog", "yolo")
SAVE_MODES = ("person", "all")


def _read_record():
    try:
        with open(PID_FILE) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_record(record):
    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    with open(PID_FILE, "w") as handle:
        json.dump(record, handle, indent=2)


def _clear_record():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _process_alive(pid):
    """Provera da proces sa tim brojem i dalje radi i da je zaista glavna
    petlja. Sam broj nije dovoljan, jer ga sistem posle gasenja moze
    dodeliti nekom drugom procesu."""
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmdline = handle.read().decode("utf-8", "replace")
    except OSError:
        return False

    return "main.py" in cmdline


def _scan_for_loop():
    """Trazi glavnu petlju medju procesima, bez obzira na to ko ju je
    pokrenuo. GPIO linija se ne zakljucava, pa se dve petlje mogu pokrenuti
    uporedo -- druga bi se otimala o kameru i kvarila merenja. Zato se
    stanje utvrdjuje pregledom procesa, a ne samo iz zapisa ove strane."""
    main_path = os.path.join(config.BASE_DIR, "main.py")

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                parts = [p for p in handle.read().decode(
                    "utf-8", "replace").split("\0") if p]
        except OSError:
            continue

        if not parts or "python" not in os.path.basename(parts[0]):
            continue
        if not any(p == main_path or p.endswith("/main.py") or p == "main.py"
                   for p in parts):
            continue

        return pid, parts

    return None, None


def _argument_after(parts, flag, default=None):
    if flag in parts:
        index = parts.index(flag)
        if index + 1 < len(parts):
            return parts[index + 1]
    return default


def status():
    record = _read_record()
    if not record or not _process_alive(record.get("pid")):
        if record:
            _clear_record()

        # Petlja mozda radi, ali je pokrenuta iz terminala ili je ostala
        # od ranijeg pokretanja servera.
        pid, parts = _scan_for_loop()
        if pid is None:
            return {"running": False, "external": False, "pid": None,
                    "detector": None, "save_mode": None, "threshold": None,
                    "started_at": None}

        return {
            "running": True,
            "external": True,
            "pid": pid,
            "detector": _argument_after(parts, "--detector", "podrazumevani"),
            "save_mode": _argument_after(parts, "--save-mode", "person"),
            "threshold": _argument_after(parts, "--threshold"),
            "started_at": None,
        }

    record["running"] = True
    record["external"] = False
    return record


def start(detector="hog", save_mode="person", threshold=None):
    """Pokrece glavnu petlju. Vraca (uspeh, poruka)."""
    current = status()
    if current["running"]:
        if current.get("external"):
            return False, (
                f"Glavna petlja vec radi (proces {current['pid']}), pokrenuta "
                f"izvan ove strane -- verovatno iz terminala. Dve petlje bi se "
                f"otimale o kameru, pa prvo zaustaviti onu.")
        return False, (f"Sistem vec radi (detektor {current['detector']}). "
                       f"Prvo ga zaustaviti.")

    if detector not in DETECTORS:
        return False, f"Nepoznat detektor: {detector}"
    if save_mode not in SAVE_MODES:
        return False, f"Nepoznat rezim cuvanja: {save_mode}"

    command = [sys.executable, "-u", os.path.join(config.BASE_DIR, "main.py"),
               "--detector", detector, "--save-mode", save_mode]
    if threshold is not None:
        command += ["--threshold", str(threshold)]

    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    log_handle = open(SYSTEM_LOG, "a")
    log_handle.write(f"\n===== pokrenuto {datetime.now():%Y-%m-%d %H:%M:%S} "
                     f"| {' '.join(command[2:])} =====\n")
    log_handle.flush()

    environment = dict(os.environ, PYTHONUNBUFFERED="1")

    try:
        process = subprocess.Popen(
            command, cwd=config.BASE_DIR, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError as error:
        log_handle.close()
        return False, f"Pokretanje nije uspelo: {error}"

    # Kratko cekanje, da se odmah vidi ako proces padne pri pokretanju
    # (zauzeta GPIO linija, nedostaje model i slicno).
    time.sleep(1.5)
    if process.poll() is not None:
        log_handle.close()
        return False, ("Proces se ugasio odmah po pokretanju. Videti dnevnik "
                       "ispod; najcesci uzrok je da glavna petlja vec radi u "
                       "terminalu i drzi GPIO liniju.")

    _write_record({
        "pid": process.pid,
        "detector": detector,
        "save_mode": save_mode,
        "threshold": threshold,
        "started_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
    })
    return True, f"Sistem pokrenut, detektor {detector}."


def stop(timeout=12.0):
    """Zaustavlja glavnu petlju. Salje SIGTERM, koji main.py hvata i uredno
    zavrsava: upisuje stanje, gasi diodu i otpusta kameru."""
    current = status()
    if not current["running"]:
        return False, "Sistem ne radi."

    pid = current["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as error:
        _clear_record()
        return False, f"Zaustavljanje nije uspelo: {error}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _process_alive(pid):
            _clear_record()
            return True, "Sistem zaustavljen."
        time.sleep(0.3)

    # Uredno gasenje ceka i na zapoceta slanja poste, pa se daje vremena;
    # tek ako ni to ne pomogne, proces se prekida.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    _clear_record()
    return True, "Sistem nije odgovorio na zahtev za gasenje, pa je prekinut."


def restart(detector, save_mode="person", threshold=None):
    if status()["running"]:
        stop()
    return start(detector, save_mode, threshold)


def tail_log(lines=20):
    try:
        with open(SYSTEM_LOG, errors="replace") as handle:
            return handle.readlines()[-lines:]
    except OSError:
        return []
