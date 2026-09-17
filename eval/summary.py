"""Sazetak rada sistema po detektoru, iz logs/events.csv.

Sluzi za brzo poredjenje dva postupka posle probnog rada:

    python3 eval/summary.py
    python3 eval/summary.py --since "2026-09-15 18:00"
"""

import argparse
import collections
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def median(rows, field):
    values = [float(r[field]) for r in rows if r.get(field)]
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None,
                        help='samo dogadjaji posle ovog trenutka, npr. "2026-09-15 18:00"')
    parser.add_argument("--csv", default=config.EVENT_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Nema evidencije: {args.csv}")

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if args.since:
        rows = [r for r in rows if r["ts"] >= args.since]

    groups = collections.defaultdict(list)
    errors = collections.Counter()
    for row in rows:
        if row["decision"] in ("person", "no_person"):
            groups[row["detector"]].append(row)
        elif row["decision"]:
            errors[row["decision"]] += 1

    if not groups:
        raise SystemExit("Nema okidanja u evidenciji.")

    print(f"{'detektor':10}{'okidanja':>10}{'osoba':>7}{'odbacila kamera':>17}"
          f"{'akvizicija':>12}{'detekcija':>11}{'odziv':>9}")
    for name, group in sorted(groups.items()):
        persons = sum(1 for r in group if r["decision"] == "person")
        rejected = 100.0 * (len(group) - persons) / len(group)
        print(f"{name:10}{len(group):>10}{persons:>7}{rejected:>16.1f}%"
              f"{median(group, 't_capture_ms'):>9.0f} ms"
              f"{median(group, 't_infer_ms'):>8.0f} ms"
              f"{median(group, 't_total_ms'):>6.0f} ms")

    for kind, count in errors.items():
        print(f"\n{kind}: {count}")

    print("\nSve vrednosti su medijane. Udeo koji je kamera odbacila racuna se "
          "\nkao okidanja bez osobe / ukupno okidanja.")


if __name__ == "__main__":
    main()
