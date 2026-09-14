"""Veb sloj sistema.

Server na ploci sastavlja stranu na zahtev pregledaca. Posto HTTP nema
stanje, svako osvezavanje iznova cita evidenciju sa diska (8.3)."""

import os
import shutil
from datetime import datetime

from flask import (Flask, abort, redirect, render_template,
                   send_from_directory, url_for)

import config
from logger import read_events, read_status

app = Flask(__name__)

# Posle koliko sekundi bez upisa stanja se smatra da sistem vise ne radi.
STALE_AFTER_S = 120


def _system_state():
    state, written_at, detector = read_status()

    alive = False
    if state == "Running" and written_at:
        try:
            age = (datetime.now()
                   - datetime.fromisoformat(written_at)).total_seconds()
            alive = age < STALE_AFTER_S
        except ValueError:
            alive = False

    if state == "Running" and not alive:
        state = "Ne odaziva se"

    return state, written_at, detector


def _summary(events):
    """Udeo okidanja koja je kamera odbacila kao lazna -- brojka kojom se
    dvostepena arhitektura potvrdjuje ili obara."""
    triggers = [e for e in events
                if e.get("decision") in ("person", "no_person")]
    with_person = [e for e in triggers if e["decision"] == "person"]

    total = len(triggers)
    persons = len(with_person)
    rejected = total - persons

    return {
        "triggers": total,
        "persons": persons,
        "rejected": rejected,
        "rejected_pct": round(100.0 * rejected / total, 1) if total else 0.0,
    }


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _timing(events):
    def column(name):
        out = []
        for event in events:
            try:
                out.append(float(event[name]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    return {
        "capture": _median(column("t_capture_ms")),
        "infer": _median(column("t_infer_ms")),
        "total": _median(column("t_total_ms")),
    }


@app.route("/")
def index():
    events = read_events()
    state, written_at, detector = _system_state()

    images = []
    if os.path.isdir(config.PHOTO_FOLDER):
        images = sorted(os.listdir(config.PHOTO_FOLDER), reverse=True)
        images = [name for name in images if name.lower().endswith(".jpg")]

    last_event = next((e for e in events if e.get("image")), None)

    _, _, free_disk = shutil.disk_usage(config.BASE_DIR)

    return render_template(
        "index.html",
        state=state,
        status_written_at=written_at,
        detector=detector,
        summary=_summary(events),
        timing=_timing(events),
        events=events[:15],
        images=images,
        total_images=len(images),
        latest=images[0] if images else None,
        last_time=last_event["ts"] if last_event else "nema detekcija",
        current_time=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        free_disk=round(free_disk / (1024 ** 3), 2),
    )


@app.route("/photos/<path:filename>")
def photos(filename):
    return send_from_directory(config.PHOTO_FOLDER, filename)


@app.route("/events.csv")
def events_csv():
    if not os.path.exists(config.EVENT_CSV):
        abort(404)
    return send_from_directory(
        os.path.dirname(config.EVENT_CSV),
        os.path.basename(config.EVENT_CSV),
        as_attachment=True,
    )


@app.route("/delete_all", methods=["POST"])
def delete_all():
    """Samo POST: na GET bi svako ucitavanje strane ili robot obrisao
    prikupljene podatke."""
    if os.path.isdir(config.PHOTO_FOLDER):
        for name in os.listdir(config.PHOTO_FOLDER):
            path = os.path.join(config.PHOTO_FOLDER, name)
            if os.path.isfile(path):
                os.remove(path)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
