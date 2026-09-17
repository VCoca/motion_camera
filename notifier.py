"""Obavestavanje korisnika elektronskom postom kada se otkrije osoba.

Tri stvari koje odredjuju izvedbu:

1. Slanje ide u zasebnoj niti. SMTP razmena traje i po nekoliko sekundi,
   a vreme odziva sistema (t3 - t0) meri se za rad, pa slanje ne sme da
   ude u taj put.

2. Primalac i prekidac se cuvaju u settings.json, koji upisuje veb strana
   a cita glavna petlja. Posto HTTP nema stanje (8.3), podesavanje zivi u
   sacuvanim podacima, a ne u procesu.

3. Pristupni podaci naloga sa koga se salje drze se odvojeno, u smtp.json
   ili u promenljivama okruzenja, i nikada se ne prikazuju na strani.
   Veb strana zadaje samo adresu primaoca.
"""

import json
import mimetypes
import os
import re
import smtplib
import threading
import time
from datetime import datetime
from email.message import EmailMessage

import config

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_threads = []
_lock = threading.Lock()


# --------------------------------------------------------------------------
# podesavanja koja zadaje korisnik na strani

DEFAULT_SETTINGS = {
    "enabled": False,
    "email": "",
    "cooldown_s": config.NOTIFY_COOLDOWN_S,
}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(config.SETTINGS_FILE) as handle:
            settings.update(json.load(handle))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(enabled, email, cooldown_s=None):
    settings = load_settings()
    settings["enabled"] = bool(enabled)
    settings["email"] = (email or "").strip()
    if cooldown_s is not None:
        settings["cooldown_s"] = max(0, int(cooldown_s))

    os.makedirs(os.path.dirname(config.SETTINGS_FILE) or ".", exist_ok=True)
    with open(config.SETTINGS_FILE, "w") as handle:
        json.dump(settings, handle, indent=2)
    return settings


def valid_email(address):
    return bool(EMAIL_PATTERN.match((address or "").strip()))


# --------------------------------------------------------------------------
# stanje vremenske zabrane, da se posta ne salje na svako okidanje

def load_state():
    try:
        with open(config.NOTIFY_STATE_FILE) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"last_sent": 0.0, "suppressed": 0, "last_error": ""}


def save_state(state):
    os.makedirs(os.path.dirname(config.NOTIFY_STATE_FILE) or ".", exist_ok=True)
    with open(config.NOTIFY_STATE_FILE, "w") as handle:
        json.dump(state, handle, indent=2)


def seconds_until_allowed(settings=None, state=None):
    """Koliko je jos sekundi do sledeceg dozvoljenog slanja."""
    settings = settings or load_settings()
    state = state or load_state()
    elapsed = time.time() - float(state.get("last_sent", 0) or 0)
    return max(0.0, float(settings.get("cooldown_s", 0)) - elapsed)


# --------------------------------------------------------------------------
# pristupni podaci naloga sa koga se salje

def load_smtp():
    """Redosled: promenljive okruzenja, pa smtp.json. Vraca None ako nije
    podeseno."""
    env = {
        "host": os.environ.get("SMTP_HOST"),
        "port": os.environ.get("SMTP_PORT"),
        "user": os.environ.get("SMTP_USER"),
        "password": os.environ.get("SMTP_PASSWORD"),
        "from": os.environ.get("SMTP_FROM"),
    }
    if env["host"] and env["user"] and env["password"]:
        env["port"] = int(env["port"] or 587)
        env["from"] = env["from"] or env["user"]
        return env

    try:
        with open(config.SMTP_FILE) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None

    if not (data.get("host") and data.get("user") and data.get("password")):
        return None

    data["port"] = int(data.get("port") or 587)
    data["from"] = data.get("from") or data["user"]
    return data


def smtp_configured():
    return load_smtp() is not None


# --------------------------------------------------------------------------
# sastavljanje i slanje

def build_message(smtp, recipient, event, image_path=None, suppressed=0):
    when = event.get("ts") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = EmailMessage()
    message["Subject"] = f"Detektovana osoba - {when}"
    message["From"] = smtp["from"]
    message["To"] = recipient

    lines = [
        "Sistem za detekciju prisustva osoba je otkrio osobu.",
        "",
        f"Vreme:      {when}",
        f"Detektor:   {event.get('detector', '-')}",
        f"Nalaza:     {event.get('n_boxes', '-')}",
    ]
    if event.get("max_score"):
        lines.append(f"Poverenje:  {event['max_score']}")
    if event.get("t_total_ms"):
        lines.append(f"Odziv:      {event['t_total_ms']} ms")
    if event.get("cpu_temp_c"):
        lines.append(f"Temperatura ploce: {event['cpu_temp_c']} C")
    if suppressed:
        lines += ["", f"Od prethodnog obavestenja bilo je jos {suppressed} "
                      f"detekcija koje nisu poslate zbog vremenske zabrane."]
    lines += ["", "Slika je u prilogu."]

    message.set_content("\n".join(lines))

    if image_path and os.path.exists(image_path):
        kind, _ = mimetypes.guess_type(image_path)
        maintype, subtype = (kind or "image/jpeg").split("/", 1)
        with open(image_path, "rb") as handle:
            message.add_attachment(handle.read(), maintype=maintype,
                                   subtype=subtype,
                                   filename=os.path.basename(image_path))
    return message


def _send(smtp, message):
    if int(smtp["port"]) == 465:
        server = smtplib.SMTP_SSL(smtp["host"], int(smtp["port"]), timeout=20)
    else:
        server = smtplib.SMTP(smtp["host"], int(smtp["port"]), timeout=20)
    try:
        server.ehlo()
        if int(smtp["port"]) != 465:
            try:
                server.starttls()
                server.ehlo()
            except smtplib.SMTPNotSupportedError:
                # Probni server bez sifrovanja; pravi nalozi ga uvek imaju.
                pass
        if smtp.get("password"):
            server.login(smtp["user"], smtp["password"])
        server.send_message(message)
    finally:
        server.quit()


def _send_in_background(smtp, recipient, event, image_path, suppressed,
                        previous_last_sent=0.0):
    state = load_state()
    try:
        message = build_message(smtp, recipient, event, image_path, suppressed)
        _send(smtp, message)
        state["last_sent"] = time.time()
        state["suppressed"] = 0
        state["last_error"] = ""
        print(f"obavestenje poslato na {recipient}", flush=True)
    except Exception as error:               # noqa: BLE001
        # Neuspeh slanja ne sme da obori nadzor prostorije. Vreme poslednjeg
        # slanja se vraca na staro: upisano je unapred da dve brze detekcije
        # ne posalju dvaput, ali ako slanje nije uspelo, prolazna greska
        # servera ne treba da potrosi ceo interval vremenske zabrane.
        state["last_sent"] = previous_last_sent
        state["last_error"] = f"{type(error).__name__}: {error}"
        print(f"obavestenje nije poslato: {state['last_error']}", flush=True)
    save_state(state)


def notify_person(event, image_path=None):
    """Salje obavestenje ako je ukljuceno, adresa ispravna i vremenska
    zabrana istekla. Vraca razlog u vidu niske, radi ispisa."""
    settings = load_settings()

    if not settings.get("enabled"):
        return "iskljuceno"

    recipient = settings.get("email", "").strip()
    if not valid_email(recipient):
        return "nema ispravne adrese"

    smtp = load_smtp()
    if smtp is None:
        return "nalog za slanje nije podesen"

    state = load_state()
    remaining = seconds_until_allowed(settings, state)
    if remaining > 0:
        state["suppressed"] = int(state.get("suppressed", 0)) + 1
        save_state(state)
        return f"vremenska zabrana, jos {remaining / 60:.1f} min"

    # Vreme poslednjeg slanja se upisuje odmah, pre nego sto nit zavrsi,
    # da dve brze detekcije ne pokrenu dva slanja uporedo.
    suppressed = int(state.get("suppressed", 0))
    previous_last_sent = float(state.get("last_sent", 0) or 0)
    state["last_sent"] = time.time()
    state["suppressed"] = 0
    save_state(state)

    thread = threading.Thread(
        target=_send_in_background,
        args=(smtp, recipient, dict(event), image_path, suppressed,
              previous_last_sent),
        daemon=True,
    )
    with _lock:
        _threads.append(thread)
    thread.start()
    return f"salje se na {recipient}"


def wait_for_pending(timeout=15.0):
    """Ceka da se zapoceta slanja zavrse pri gasenju sistema."""
    with _lock:
        threads = [t for t in _threads if t.is_alive()]
    deadline = time.time() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.time()))


# --------------------------------------------------------------------------
# provera podesavanja iz komandne linije:  python3 notifier.py --test

def _self_test():
    """Salje probnu poruku na adresu zadatu na veb strani, zaobilazeci
    vremensku zabranu. Sluzi za proveru naloga posle popunjavanja
    smtp.json."""
    settings = load_settings()
    recipient = settings.get("email", "").strip()

    print(f"Primalac:     {recipient or '(nije zadat na veb strani)'}")
    print(f"Obavestavanje: {'ukljuceno' if settings.get('enabled') else 'iskljuceno'}")

    if not valid_email(recipient):
        raise SystemExit("Adresa primaoca nije zadata ili nije ispravna. "
                         "Uneti je na veb strani.")

    smtp = load_smtp()
    if smtp is None:
        raise SystemExit(f"Nalog za slanje nije podesen. Popuniti "
                         f"{config.SMTP_FILE} po uzoru na smtp.example.json.")

    print(f"Server:       {smtp['host']}:{smtp['port']}")
    print(f"Nalog:        {smtp['user']}")
    print(f"Salje se sa:  {smtp['from']}")
    print("\nSaljem probnu poruku...")

    event = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detector": "proba",
        "n_boxes": "0",
    }
    message = build_message(smtp, recipient, event)
    del message["Subject"]
    message["Subject"] = "Probna poruka - sistem za detekciju prisustva"

    try:
        _send(smtp, message)
    except smtplib.SMTPAuthenticationError:
        raise SystemExit(
            "Prijava odbijena. Kod Gmail naloga to je skoro uvek jedno od "
            "dvoje:\n"
            "  - koriscena je obicna lozinka umesto lozinke za aplikaciju\n"
            "  - dvostepena provera nije ukljucena, pa lozinke za aplikaciju "
            "nema"
        )
    except Exception as error:               # noqa: BLE001
        raise SystemExit(f"Slanje nije uspelo: {type(error).__name__}: {error}")

    print(f"Poslato na {recipient}. Proveri sanduce, i fasciklu za nezeljenu "
          f"postu.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
        print("Provera podesavanja:  python3 notifier.py --test")
