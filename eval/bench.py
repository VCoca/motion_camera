"""Vrednovanje van linije: oba detektora nad istim skupom slika.

Poredjenje je posteno samo ako oba postupka dobiju isti ulaz, pa se ne meri
u toku rada sistema nego nad snimljenim skupom. Skript upisuje sve nalaze
sa merama poverenja, bez praga, da bi se kriva preciznosti i odziva mogla
nacrtati iz jednog prolaza.

Pokretanje:
    python3 eval/bench.py                    oba detektora
    python3 eval/bench.py --detector hog
    python3 eval/bench.py --repeat 3         ponovljena merenja vremena
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

import config  # noqa: E402
from detectors import AVAILABLE, create_detector  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(EVAL_DIR, "dataset", "images")
RESULTS_DIR = os.path.join(config.BASE_DIR, "results")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_images(directory):
    if not os.path.isdir(directory):
        raise SystemExit(
            f"Nema mape sa slikama: {directory}\n"
            f"Snimiti skup sa: python3 main.py --save-mode all\n"
            f"pa slike prebaciti u {directory}"
        )
    names = [n for n in sorted(os.listdir(directory))
             if n.lower().endswith(IMAGE_SUFFIXES)]
    if not names:
        raise SystemExit(f"Mapa {directory} je prazna.")
    return names


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def run_detector(name, image_names, repeat):
    detector = create_detector(name)
    print(f"\n{detector.describe()}")
    print(f"{len(image_names)} slika, {repeat} prolaz(a) po slici")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detections_path = os.path.join(RESULTS_DIR, f"detections_{name}.csv")
    timing_path = os.path.join(RESULTS_DIR, f"timing_{name}.csv")

    all_times = []

    with open(detections_path, "w", newline="") as det_file, \
            open(timing_path, "w", newline="") as time_file:

        det_writer = csv.writer(det_file)
        det_writer.writerow(["image", "x", "y", "w", "h", "score"])

        time_writer = csv.writer(time_file)
        time_writer.writerow(["image", "width", "height", "infer_ms"])

        for index, image_name in enumerate(image_names, start=1):
            frame = cv2.imread(os.path.join(IMAGES_DIR, image_name))
            if frame is None:
                print(f"  preskocena neispravna slika: {image_name}")
                continue

            height, width = frame.shape[:2]

            times = []
            detections = []
            for _ in range(repeat):
                detections, elapsed_ms = detector.detect(frame)
                times.append(elapsed_ms)

            # Za merenje se uzima najbolji prolaz, jer je on najmanje
            # zagadjen radom ostalih procesa na ploci.
            best_ms = min(times)
            all_times.append(best_ms)
            time_writer.writerow([image_name, width, height, f"{best_ms:.2f}"])

            for det in detections:
                det_writer.writerow([image_name, det.x, det.y, det.w, det.h,
                                     f"{det.score:.6f}"])

            if index % 10 == 0 or index == len(image_names):
                print(f"  {index}/{len(image_names)}")

    summary = {
        "detector": name,
        "opis": detector.describe(),
        "slika": len(all_times),
        "median_ms": round(statistics.median(all_times), 1) if all_times else None,
        "p95_ms": round(percentile(all_times, 0.95), 1) if all_times else None,
        "min_ms": round(min(all_times), 1) if all_times else None,
        "max_ms": round(max(all_times), 1) if all_times else None,
    }

    print(f"  median {summary['median_ms']} ms · "
          f"95. percentil {summary['p95_ms']} ms · "
          f"raspon {summary['min_ms']}–{summary['max_ms']} ms")
    print(f"  nalazi -> {detections_path}")
    print(f"  vremena -> {timing_path}")

    return summary


def main():
    global IMAGES_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=AVAILABLE, default=None,
                        help="samo jedan detektor umesto oba")
    parser.add_argument("--repeat", type=int, default=3,
                        help="broj prolaza po slici za merenje vremena")
    parser.add_argument("--images", default=IMAGES_DIR,
                        help="mapa sa slikama skupa")
    args = parser.parse_args()

    IMAGES_DIR = args.images

    image_names = list_images(IMAGES_DIR)
    names = [args.detector] if args.detector else list(AVAILABLE)

    summaries = [run_detector(name, image_names, args.repeat) for name in names]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timing_summary = os.path.join(RESULTS_DIR, "timing_summary.csv")
    with open(timing_summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSazetak vremena -> {timing_summary}")
    print("Sledeci korak: python3 eval/metrics.py")


if __name__ == "__main__":
    main()
