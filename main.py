"""Glavna petlja sistema za detekciju prisustva osoba.

Prvi stepen je PIR senzor koji neprekidno nadgleda prostor, drugi je kamera
sa postupkom otkrivanja osoba na slici. Po okidanju se belezi cetiri
trenutka, iz kojih slede sva vremena potrebna za merenja u radu:

    t0  uzlazna ivica na GPIO liniji
    t1  kadar u memoriji          -> t1 - t0 = akvizicija
    t2  odluka detektora          -> t2 - t1 = zakljucivanje
    t3  slika i zapis na disku    -> t3 - t0 = odziv sistema

Pokretanje:
    python3 main.py                          radni rezim, detektor iz config.py
    python3 main.py --detector yolo
    python3 main.py --save-mode all          rezim prikupljanja skupa podataka
"""

import argparse
import signal
import sys
import time

import config
from camera import capture_frame, draw_boxes, release_camera, save_frame
from detectors import AVAILABLE, create_detector, filter_by_score
from led import led_off, led_on, led_close
from logger import log_event, write_status
from sensor import PirSensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=AVAILABLE, default=None,
                        help="postupak za otkrivanje osoba na slici")
    parser.add_argument("--save-mode", choices=("person", "all"), default=None,
                        help="'all' cuva svaki kadar posle okidanja, bez "
                             "iscrtanih okvira, za skup podataka")
    parser.add_argument("--threshold", type=float, default=None,
                        help="prag mere poverenja (radna tacka)")
    parser.add_argument("--warmup", type=float, default=None,
                        help="vreme inicijalizacije senzora u sekundama")
    args = parser.parse_args()

    if args.save_mode is not None:
        config.SAVE_MODE = args.save_mode
    if args.threshold is not None:
        config.SCORE_THRESHOLD = args.threshold
    if args.warmup is not None:
        config.WARMUP_S = args.warmup

    return args


def handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)

    detector = create_detector(args.detector)
    sensor = PirSensor()

    print(f"Detektor:     {detector.describe()}")
    print(f"Rezim cuvanja: {config.SAVE_MODE}")
    print(f"Prag:          {config.SCORE_THRESHOLD}")
    print(f"Inicijalizacija senzora: {sensor.warmup_remaining():.0f} s "
          f"(okidanja u tom vremenu se odbacuju)")
    print("Cekam pokret...\n")

    write_status("Running", detector.name)

    try:
        while True:
            # Kratko vreme cekanja da bi prekid tastaturom bio odziv.
            t0 = sensor.wait_for_motion(timeout=1.0)
            if t0 is None:
                continue

            led_on()

            frame, capture_ms = capture_frame()

            if frame is None:
                print("Greska kamere.")
                log_event(detector=detector.name, t_capture_ms=capture_ms,
                          decision="camera_error")
                led_off()
                sensor.wait_for_no_motion(timeout=10)
                continue

            height, width = frame.shape[:2]

            detections, infer_ms = detector.detect(frame)
            detections = filter_by_score(detections, config.SCORE_THRESHOLD)
            found = len(detections) > 0

            image = ""
            if config.SAVE_MODE == "all":
                # Za skup podataka se cuva neizmenjen kadar: iscrtani okviri
                # bi kvarili kasnije oznacavanje.
                image = save_frame(frame, prefix="frame")
            elif found:
                draw_boxes(frame, detections)
                image = save_frame(frame, prefix="person")

            total_ms = (time.monotonic() - t0) * 1000.0
            max_score = detections[0].score if found else None

            log_event(
                detector=detector.name,
                width=width, height=height,
                t_capture_ms=capture_ms,
                t_infer_ms=infer_ms,
                t_total_ms=total_ms,
                n_boxes=len(detections),
                max_score=max_score,
                decision="person" if found else "no_person",
                image=image,
            )

            print(f"okidanje | akvizicija {capture_ms:6.1f} ms | "
                  f"detekcija {infer_ms:6.1f} ms | ukupno {total_ms:6.1f} ms | "
                  f"{'osoba: ' + str(len(detections)) if found else 'nema osobe'}"
                  f"{' | ' + image if image else ''}")

            sensor.wait_for_no_motion(timeout=30)
            led_off()
            time.sleep(config.COOLDOWN)

    except KeyboardInterrupt:
        print("\nZaustavljanje...")

    finally:
        if sensor.discarded_in_warmup:
            print(f"Odbaceno okidanja u inicijalizaciji: "
                  f"{sensor.discarded_in_warmup}")
        write_status("Stopped", detector.name)
        led_off()
        led_close()
        release_camera()
        sensor.close()


if __name__ == "__main__":
    sys.exit(main())
