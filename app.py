"""Veb sloj sistema.

Server na ploci sastavlja stranu na zahtev pregledaca. Posto HTTP nema
stanje, svako osvezavanje iznova cita evidenciju sa diska (8.3)."""

import os
import shutil
from datetime import datetime

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_from_directory, url_for)

import config
import notifier
import supervisor
from logger import read_events, read_status

app = Flask(__name__)
# Kljuc sluzi samo za kratke poruke korisniku posle cuvanja podesavanja.
app.secret_key = os.environ.get("FLASK_SECRET", "motion-camera")

# Posle koliko sekundi bez upisa stanja se smatra da sistem vise ne radi.
STALE_AFTER_S = 120


def _system_state():
    """Stanje se utvrdjuje iz samog procesa, ne iz onoga sto je zapisao:
    proces koji je pao ne stigne da upise da vise ne radi."""
    process = supervisor.status()
    _, written_at, detector = read_status()

    if process["running"]:
        return "Radi", written_at, process["detector"]

    return "Zaustavljen", written_at, detector


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

    process = supervisor.status()

    notify_settings = notifier.load_settings()
    notify_state = notifier.load_state()

    last_sent = ""
    if notify_state.get("last_sent"):
        last_sent = datetime.fromtimestamp(
            notify_state["last_sent"]).strftime("%d.%m.%Y %H:%M:%S")

    notify = {
        "enabled": notify_settings.get("enabled", False),
        "email": notify_settings.get("email", ""),
        "cooldown_min": round(notify_settings.get("cooldown_s", 300) / 60, 1),
        "smtp_ok": notifier.smtp_configured(),
        "last_sent": last_sent,
        "suppressed": notify_state.get("suppressed", 0),
        "last_error": notify_state.get("last_error", ""),
        "wait_min": round(
            notifier.seconds_until_allowed(notify_settings, notify_state) / 60, 1),
    }

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
        notify=notify,
        process=process,
        detectors=supervisor.DETECTORS,
        system_log=supervisor.tail_log(14),
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


@app.route("/settings", methods=["POST"])
def settings():
    """Prima adresu primaoca, prekidac i vremensku zabranu sa strane."""
    email = request.form.get("email", "").strip()
    enabled = request.form.get("enabled") == "on"

    try:
        cooldown_s = int(float(request.form.get("cooldown_min", 5)) * 60)
    except ValueError:
        cooldown_s = config.NOTIFY_COOLDOWN_S

    if enabled and not notifier.valid_email(email):
        flash("Adresa nije ispravna, obavestavanje nije ukljuceno.", "bad")
        enabled = False

    notifier.save_settings(enabled, email, cooldown_s)

    if enabled and not notifier.smtp_configured():
        flash("Sacuvano, ali nalog za slanje nije podesen "
              "(videti smtp.example.json).", "bad")
    else:
        flash("Podesavanja sacuvana.", "ok")

    return redirect(url_for("index"))


@app.route("/system/start", methods=["POST"])
def system_start():
    detector = request.form.get("detector", "hog")
    save_mode = request.form.get("save_mode", "person")

    threshold = request.form.get("threshold", "").strip()
    try:
        threshold = float(threshold) if threshold else None
    except ValueError:
        threshold = None

    ok, message = supervisor.start(detector, save_mode, threshold)
    flash(message, "ok" if ok else "bad")
    return redirect(url_for("index"))


@app.route("/system/stop", methods=["POST"])
def system_stop():
    ok, message = supervisor.stop()
    flash(message, "ok" if ok else "bad")
    return redirect(url_for("index"))


@app.route("/system/restart", methods=["POST"])
def system_restart():
    """Promena detektora u toku rada: petlja se zaustavi i podigne sa
    novim postupkom, jer se detektor bira pri pokretanju."""
    detector = request.form.get("detector", "hog")
    save_mode = request.form.get("save_mode", "person")

    ok, message = supervisor.restart(detector, save_mode)
    flash(message, "ok" if ok else "bad")
    return redirect(url_for("index"))


@app.route("/test_mail", methods=["POST"])
def test_mail():
    """Probna poruka, bez cekanja da neko prodje ispred senzora."""
    settings = notifier.load_settings()
    recipient = settings.get("email", "").strip()

    if not notifier.valid_email(recipient):
        flash("Prvo uneti ispravnu adresu primaoca.", "bad")
        return redirect(url_for("index"))

    smtp = notifier.load_smtp()
    if smtp is None:
        flash("Nalog za slanje nije podesen (smtp.json).", "bad")
        return redirect(url_for("index"))

    event = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "detector": "proba", "n_boxes": "0"}
    message = notifier.build_message(smtp, recipient, event)
    del message["Subject"]
    message["Subject"] = "Probna poruka - sistem za detekciju prisustva"

    try:
        notifier._send(smtp, message)
    except Exception as error:               # noqa: BLE001
        flash(f"Slanje nije uspelo: {type(error).__name__}: {error}", "bad")
        return redirect(url_for("index"))

    flash(f"Probna poruka poslata na {recipient}.", "ok")
    return redirect(url_for("index"))


@app.route("/clear_events", methods=["POST"])
def clear_events():
    """Brise evidenciju merenja. Koristi se pre pocetka novog merenja, da
    se prolazi dva detektora ne pomesaju sa probama."""
    try:
        os.remove(config.EVENT_CSV)
        flash("Evidencija obrisana.", "ok")
    except OSError:
        flash("Evidencija ne postoji.", "bad")
    return redirect(url_for("index"))


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
