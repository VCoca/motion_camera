"""Indikatorska dioda na GPIO liniji.

Dioda je vezana redno sa zastitnim otpornikom od 270 ohma (proracun u
poglavlju 6 istrazivackog rada), pa je radna struja oko 5 mA, dovoljno
ispod granice izlaznog stepena od 16 mA."""

from gpiozero import LED

import config

led = LED(config.LED_PIN)


def led_on():
    led.on()


def led_off():
    led.off()


def blink(duration=1.0):
    from time import sleep
    led.on()
    sleep(duration)
    led.off()


def led_close():
    led.close()
