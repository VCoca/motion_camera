"""Podela skupa na deo za podesavanje i deo za vrednovanje (odeljak 6.2 rada).

Prag poverenja i radna tacka biraju se gledajuci podatke. Ako bi se birali
nad celim skupom, a zatim se nad istim tim skupom merio rezultat, merilo bi
se koliko su vrednosti pogodjene bas za te slike, a ne koliko postupak valja.
Zato se skup deli: na jednom delu se bira, na drugom meri, i ta dva dela se
ne dodiruju.

Podela NIJE nasumicna. Kadrovi snimljeni u razmaku od nekoliko sekundi skoro
su iste slike; nasumicna podela stavila bi gotovo isti kadar na obe strane,
sto je ista greska kao da podele nema, samo sakrivena. Zato se deli po
vremenu snimanja, i to unutar svakog scenarija zasebno: prvih TUNE_PER_SESSION
kadrova svake sesije ide u deo za podesavanje, ostatak u deo za vrednovanje.
Time su svi uslovi snimanja zastupljeni u oba dela, a susedni kadrovi ostaju
zajedno.

Podela se upisuje u datoteku i od tada se ne menja. Ponovno pokretanje cita
zatecenu podelu; za novu je potrebno izricito --rebuild. Bez toga bi se
podela mogla nehotice promeniti posle izbora praga, cime bi ceo postupak
izgubio smisao.

Pokretanje:
    python3 eval/split.py                 napravi ili prikazi podelu
    python3 eval/split.py --rebuild       napravi je iznova
    python3 eval/split.py --tune-per-session 8
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(EVAL_DIR, "dataset", "images")
SPLIT_CSV = os.path.join(EVAL_DIR, "dataset", "split.csv")
FRAMES_CSV = config.FRAMES_CSV

TUNE = "podesavanje"
EVAL = "vrednovanje"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_images(images_dir):
    return [n for n in sorted(os.listdir(images_dir))
            if n.lower().endswith(IMAGE_SUFFIXES)]


def read_frames(frames_csv):
    """Vraca {ime_slike: (scenario, ts)} iz evidencije prikupljanja."""
    meta = {}
    if not os.path.exists(frames_csv):
        return meta
    with open(frames_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("frame_id") or "").strip()
            if name:
                meta[name] = ((row.get("scenario") or "").strip(),
                              (row.get("ts") or "").strip())
    return meta


def build(images_dir=IMAGES_DIR, frames_csv=FRAMES_CSV,
          tune_per_session=6, path=SPLIT_CSV):
    """Pravi podelu i upisuje je. Vraca {ime: deo}."""
    names = list_images(images_dir)
    if not names:
        raise SystemExit(f"Mapa {images_dir} je prazna.")

    meta = read_frames(frames_csv)
    missing = [n for n in names if n not in meta]
    if missing:
        print(f"Upozorenje: {len(missing)} slika nema red u "
              f"{os.path.basename(frames_csv)}, pa im je scenario nepoznat. "
              f"Svrstane su u jednu sesiju po imenu datoteke.")

    sessions = defaultdict(list)
    for name in names:
        scenario, ts = meta.get(name, ("", ""))
        # Ime datoteke sadrzi vremensku oznaku, pa je pouzdan kljuc za
        # redjanje i kada evidencija nedostaje.
        sessions[scenario or "(nepoznat)"].append((ts or name, name))

    assignment = {}
    for scenario in sorted(sessions):
        ordered = [n for _, n in sorted(sessions[scenario])]
        for index, name in enumerate(ordered):
            assignment[name] = TUNE if index < tune_per_session else EVAL

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "scenario", "deo"])
        for name in names:
            writer.writerow([name, meta.get(name, ("", ""))[0], assignment[name]])

    return assignment


def load(path=SPLIT_CSV):
    """Cita zatecenu podelu. Vraca {ime: deo} ili prazan recnik ako je nema."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["frame_id"]] = row["deo"]
    return out


def get(images_dir=IMAGES_DIR, frames_csv=FRAMES_CSV, tune_per_session=6,
        path=SPLIT_CSV, rebuild=False, quiet=False):
    """Vraca podelu; pravi je samo ako je nema ili ako se izricito trazi.

    Ako skupu pridju nove slike, podela se dopunjuje bez diranja postojecih
    dodela, da izbor praga ostane vezan za iste slike na kojima je nastao."""
    assignment = {} if rebuild else load(path)
    names = list_images(images_dir)

    if not assignment:
        assignment = build(images_dir, frames_csv, tune_per_session, path)
        if not quiet:
            print(f"Napravljena podela -> {path}")
    else:
        new = [n for n in names if n not in assignment]
        if new:
            if not quiet:
                print(f"{len(new)} novih slika nije bilo u podeli; "
                      f"dodeljuju se delu za vrednovanje, a postojece dodele "
                      f"ostaju netaknute.")
            meta = read_frames(frames_csv)
            for name in new:
                assignment[name] = EVAL
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_id", "scenario", "deo"])
                for name in names:
                    writer.writerow([name, meta.get(name, ("", ""))[0],
                                     assignment[name]])

    return {n: assignment[n] for n in names if n in assignment}


def summary(assignment, frames_csv=FRAMES_CSV):
    meta = read_frames(frames_csv)
    per = defaultdict(lambda: [0, 0])
    for name, part in assignment.items():
        scenario = meta.get(name, ("", ""))[0] or "(nepoznat)"
        per[scenario][0 if part == TUNE else 1] += 1
    return per


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-per-session", type=int, default=6,
                        help="koliko prvih kadrova svake sesije ide u deo "
                             "za podesavanje")
    parser.add_argument("--rebuild", action="store_true",
                        help="napravi podelu iznova, i kada vec postoji")
    parser.add_argument("--images", default=IMAGES_DIR)
    parser.add_argument("--frames", default=FRAMES_CSV)
    parser.add_argument("--out", default=SPLIT_CSV)
    args = parser.parse_args()

    assignment = get(args.images, args.frames, args.tune_per_session,
                     args.out, args.rebuild)

    print(f"\n{'scenario':<16} {'podesavanje':>12} {'vrednovanje':>12}")
    total = [0, 0]
    for scenario, (a, b) in sorted(summary(assignment, args.frames).items()):
        print(f"{scenario:<16} {a:>12} {b:>12}")
        total[0] += a
        total[1] += b
    print(f"{'ukupno':<16} {total[0]:>12} {total[1]:>12}")
    print(f"\nPodela -> {args.out}")


if __name__ == "__main__":
    main()
