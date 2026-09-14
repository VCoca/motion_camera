"""Citanje modula HC-SR501 preko GPIO linije.

Linija se trazi kao ulaz sa otkrivanjem ivice, pa program ceka na dogadjaj
umesto da u petlji ispituje stanje (3.5 istrazivackog rada). Trenutak ivice
se pamti u samom povratnom pozivu, jer je to polazna tacka t0 za merenje
vremena odziva sistema."""

import threading
import time

from gpiozero import DigitalInputDevice

import config


class PirSensor:

    def __init__(self, pin=None, warmup_s=None):
        pin = config.GPIO_PIN if pin is None else pin
        self.warmup_s = config.WARMUP_S if warmup_s is None else warmup_s

        # pull_up=False ukljucuje ugradjeni otpornik spusten na masu, kao
        # zastitu za period pre nego sto modul preuzme upravljanje linijom.
        self._device = DigitalInputDevice(pin, pull_up=False)

        self._event = threading.Event()
        self._lock = threading.Lock()
        self._t0 = None

        self._started_at = time.monotonic()
        self.discarded_in_warmup = 0

        self._device.when_activated = self._on_rising_edge

    def _on_rising_edge(self):
        t0 = time.monotonic()

        # Posle ukljucenja napajanja modul do oko jednog minuta daje lazne
        # impulse (4.6.2), pa se ta okidanja odbacuju.
        if t0 - self._started_at < self.warmup_s:
            self.discarded_in_warmup += 1
            return

        with self._lock:
            self._t0 = t0
        self._event.set()

    @property
    def in_warmup(self):
        return (time.monotonic() - self._started_at) < self.warmup_s

    def warmup_remaining(self):
        return max(0.0, self.warmup_s - (time.monotonic() - self._started_at))

    def wait_for_motion(self, timeout=None):
        """Blokira do okidanja. Vraca trenutak ivice (time.monotonic) ili
        None ako je isteklo vreme."""
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        with self._lock:
            return self._t0

    def clear_pending(self):
        """Odbacuje ivicu koja je nastala dok je sistem obradjivao prethodno
        okidanje. Bez toga bi sledeci prolaz dobio zastareo trenutak t0, pa
        bi izmereni odziv sadrzao i vreme cekanja u redu, a ne samo odziv
        sistema. Uz to je isti neprekidni pokret inace obradjivan dvaput."""
        self._event.clear()
        with self._lock:
            self._t0 = None

    def wait_for_no_motion(self, timeout=None):
        return self._device.wait_for_inactive(timeout)

    @property
    def is_active(self):
        return bool(self._device.value)

    def close(self):
        self._device.close()
