"""Mere vrednovanja detekcije: IoU, preciznost, odziv, AP.

Postupak je onaj iz odeljka 7.6 istrazivackog rada. Nalazi se redjaju po
meri poverenja naniže i uparuju sa tacnim okvirima pri pragu IoU; svaki
tacan okvir moze biti upareni najvise jednom, ostalo su lazno pozitivni.

Kljucni izlaz je poredjenje pri UPOREDIVOM ODZIVU: za zadati ciljni odziv
trazi se prag poverenja koji ga kod svakog postupka daje, pa se tek tada
porede preciznosti. Poredjenje pri podrazumevanim pragovima nije posteno,
jer skorovi dva postupka nisu ista velicina.

Mera se racuna u dva koraka, po odeljcima 6.2 i 6.6 rada:

    1. na delu skupa ZA PODESAVANJE trazi se prag pri kome odziv dostize
       ciljnu vrednost;
    2. taj prag se zatim primenjuje NEPROMENJEN na deo ZA VREDNOVANJE, i
       tek te vrednosti se prijavljuju.

Kada bi se prag birao i merio nad istim slikama, merilo bi se koliko je prag
pogodjen bas za njih, a ne koliko postupak valja. Podelu pravi eval/split.py.

Pokretanje:
    python3 eval/metrics.py
    python3 eval/metrics.py --iou 0.5 --target-recall 0.90
    python3 eval/metrics.py --no-split      sve slike, samo za proveru
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import split as dataset_split  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(EVAL_DIR, "dataset", "images")
LABELS_DIR = os.path.join(EVAL_DIR, "dataset", "labels")
RESULTS_DIR = os.path.join(config.BASE_DIR, "results")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# ucitavanje

def load_ground_truth(images_dir, labels_dir, min_height=40.0):
    """Oznake u YOLO obliku: klasa cx cy w h, sve relativno prema dimenzijama
    slike. Slika bez oznaka (nema datoteke ili je prazna) je negativan primer
    i bez njih nema preciznosti.

    Vraca {ime: (okviri, zanemareni_okviri)}. Zanemaren je okvir cija je
    klasa razlicita od nule (tako se oznacavaju osobe zaklonjene vise od
    polovine) i okvir nizi od min_height piksela. Po pravilima iz odeljka
    6.3 rada, takvi okviri ne ulaze ni u odziv ni u preciznost: ne broje se
    kao propustena detekcija, a nalaz koji padne na njih se odbacuje."""
    import cv2

    truth = {}
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith(IMAGE_SUFFIXES):
            continue

        image = cv2.imread(os.path.join(images_dir, name))
        if image is None:
            continue
        height, width = image.shape[:2]

        boxes, ignored = [], []
        label_path = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    cx, cy, bw, bh = (float(p) for p in parts[1:5])
                    box = (
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        bw * width,
                        bh * height,
                    )
                    if class_id != 0 or box[3] < min_height:
                        ignored.append(box)
                    else:
                        boxes.append(box)
        truth[name] = (boxes, ignored)

    return truth


def load_detections(path):
    detections = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            detections.append((
                row["image"],
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
                float(row["score"]),
            ))
    return detections


# --------------------------------------------------------------------------
# mere

def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_detection(box, area_box):
    """Deo POVRSINE NALAZA koji pada u dati okvir. Za zanemarene okvire se
    koristi ova mera, a ne IoU: nalaz koji pokriva samo deo zaklonjene osobe
    ima mali IoU, ali svakako ne treba da se broji kao lazan."""
    ax, ay, aw, ah = box
    bx, by, bw, bh = area_box

    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    own = aw * ah
    return intersection / own if own > 0 else 0.0


def match(truth, detections, iou_threshold, ignore_overlap=0.5):
    """Uparuje nalaze sa tacnim okvirima, od najveceg skora naniže.

    Vraca listu (skor, ishod) u tom redosledu, gde je ishod "tp", "fp" ili
    "odbacen". Svaki tacan okvir moze se upariti najvise jednom; nalaz koji
    vecim delom svoje povrsine padne na zanemaren okvir ne broji se nikako."""
    detections = sorted(detections, key=lambda d: d[5], reverse=True)
    matched = defaultdict(set)
    outcomes = []

    for image, x, y, w, h, score in detections:
        boxes, ignored = truth.get(image, ([], []))

        best_iou, best_index = 0.0, -1
        for index, gt_box in enumerate(boxes):
            if index in matched[image]:
                continue
            value = iou((x, y, w, h), gt_box)
            if value > best_iou:
                best_iou, best_index = value, index

        if best_iou >= iou_threshold and best_index >= 0:
            matched[image].add(best_index)
            outcomes.append((score, "tp"))
        elif any(intersection_over_detection((x, y, w, h), box)
                 >= ignore_overlap for box in ignored):
            # Nalaz je pao na zanemaren okvir: ne broji se nikako.
            outcomes.append((score, "odbacen"))
        else:
            outcomes.append((score, "fp"))

    return outcomes


def count_gt(truth):
    return sum(len(boxes) for boxes, _ in truth.values())


def evaluate(truth, detections, iou_threshold, ignore_overlap=0.5):
    """Vraca (tacke, broj_okvira, odbacenih). Tacka je (prag, odziv,
    preciznost, lazno_pozitivnih), po opadajucem skoru."""
    total_gt = count_gt(truth)
    if total_gt == 0:
        raise SystemExit("Ovaj deo skupa nema nijedan oznacen okvir.")

    points = []
    true_positives = false_positives = discarded = 0

    for score, outcome in match(truth, detections, iou_threshold,
                                ignore_overlap):
        if outcome == "odbacen":
            discarded += 1
            continue
        if outcome == "tp":
            true_positives += 1
        else:
            false_positives += 1
        recall = true_positives / total_gt
        precision = true_positives / (true_positives + false_positives)
        points.append((score, recall, precision, false_positives))

    return points, total_gt, discarded


def at_threshold(truth, detections, iou_threshold, threshold,
                 ignore_overlap=0.5):
    """Primenjuje UNAPRED ZADAT prag i vraca (tp, fp, odbacenih, broj_okvira).

    Prag dolazi sa dela skupa za podesavanje i ovde se vise ne bira, pa se
    vrednosti smeju prijaviti kao rezultat."""
    total_gt = count_gt(truth)
    kept = [d for d in detections if d[5] >= threshold]

    tp = fp = discarded = 0
    for _, outcome in match(truth, kept, iou_threshold, ignore_overlap):
        if outcome == "tp":
            tp += 1
        elif outcome == "fp":
            fp += 1
        else:
            discarded += 1

    return tp, fp, discarded, total_gt


def wilson(successes, total, z=1.96):
    """Vilsonov interval poverenja za udeo. Pouzdaniji od uobicajenog
    normalnog priblizenja kada je udeo blizu nule ili jedinice, a upravo
    tamo se radna tacka i bira (odziv 0,90)."""
    if total <= 0:
        return (None, None)
    p = successes / total
    d = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return (max(0.0, center - half), min(1.0, center + half))


def average_precision(points):
    """Povrsina ispod krive, sa interpolacijom u svim tackama."""
    recalls = [0.0] + [p[1] for p in points] + [1.0]
    precisions = [0.0] + [p[2] for p in points] + [0.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    area = 0.0
    for i in range(len(recalls) - 1):
        area += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    return area


def operating_point(points, target_recall):
    """Prva tacka na krivoj koja dostize ciljni odziv. Vraca
    (prag, odziv, preciznost, lazno_pozitivnih) ili None ako postupak taj
    odziv uopste ne dostize."""
    for point in points:
        if point[1] >= target_recall:
            return point
    return None


# --------------------------------------------------------------------------
# crtanje

def write_pr_svg(curves, path, iou_threshold, target_recall, marks=()):
    """Kriva preciznosti i odziva kao SVG, da se moze uneti u rad kao
    vektorska slika."""
    width, height = 560, 420
    left, right, top, bottom = 70, 30, 30, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    colors = ["#A8460C", "#12606B", "#4A4A4A"]

    def sx(value):
        return left + value * plot_w

    def sy(value):
        return top + (1.0 - value) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
    ]

    for step in range(6):
        value = step / 5.0
        x, y = sx(value), sy(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                     f'y2="{top + plot_h}" stroke="#E4E4E4" stroke-width="1"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
                     f'y2="{y:.1f}" stroke="#E4E4E4" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" '
                     f'text-anchor="middle" fill="#333">{value:.1f}</text>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" '
                     f'text-anchor="end" fill="#333">{value:.1f}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
                 f'y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" '
                 f'y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    tx = sx(target_recall)
    parts.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" '
                 f'y2="{top + plot_h}" stroke="#999" stroke-width="1" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{tx + 5:.1f}" y="{top + 14}" fill="#777">'
                 f'циљни одзив {target_recall:.2f}</text>')

    for index, (name, points, ap) in enumerate(curves):
        color = colors[index % len(colors)]
        coords = " ".join(f"{sx(pt[1]):.1f},{sy(pt[2]):.1f}" for pt in points)
        if coords:
            parts.append(f'<polyline points="{coords}" fill="none" '
                         f'stroke="{color}" stroke-width="2"/>')
        legend_y = top + 20 + index * 18
        parts.append(f'<rect x="{left + plot_w - 150}" y="{legend_y - 9}" '
                     f'width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{left + plot_w - 132}" y="{legend_y + 1}" '
                     f'fill="#333">{name} · AP {ap:.3f}</text>')

    for index, (_, recall, precision) in enumerate(marks):
        color = colors[index % len(colors)]
        parts.append(f'<circle cx="{sx(recall):.1f}" cy="{sy(precision):.1f}" '
                     f'r="4.5" fill="#FFFFFF" stroke="{color}" '
                     f'stroke-width="2"/>')

    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 22}" '
                 f'text-anchor="middle" fill="#333">одзив</text>')
    parts.append(f'<text x="18" y="{top + plot_h / 2:.0f}" '
                 f'text-anchor="middle" fill="#333" '
                 f'transform="rotate(-90 18 {top + plot_h / 2:.0f})">'
                 f'прецизност</text>')
    parts.append(f'<text x="{left}" y="18" fill="#666">'
                 f'праг IoU = {iou_threshold:.2f}</text>')
    parts.append("</svg>")

    with open(path, "w") as handle:
        handle.write("\n".join(parts))


# --------------------------------------------------------------------------

def split_truth(truth, assignment, part):
    return {n: v for n, v in truth.items() if assignment.get(n) == part}


def split_detections(detections, assignment, part):
    return [d for d in detections if assignment.get(d[0]) == part]


def describe(truth, label):
    positives = sum(1 for boxes, _ in truth.values() if boxes)
    total_gt = count_gt(truth)
    ignored = sum(len(ig) for _, ig in truth.values())
    print(f"{label}: {len(truth)} slika, {positives} sa osobom, "
          f"{len(truth) - positives} bez osobe, {total_gt} okvira"
          + (f", {ignored} zanemarenih" if ignored else ""))
    return total_gt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iou", type=float, default=0.5,
                        help="prag odnosa preseka i unije za uparivanje")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="odziv pri kome se porede preciznosti")
    parser.add_argument("--images", default=IMAGES_DIR)
    parser.add_argument("--labels", default=LABELS_DIR)
    parser.add_argument("--min-height", type=float, default=40.0,
                        help="okviri nizi od ovoliko piksela se zanemaruju "
                             "(odeljak 6.3 rada)")
    parser.add_argument("--tune-per-session", type=int, default=6,
                        help="koliko prvih kadrova svake sesije ide u deo za "
                             "podesavanje")
    parser.add_argument("--rebuild-split", action="store_true",
                        help="napravi podelu iznova; inace se koristi zatecena")
    parser.add_argument("--no-split", action="store_true",
                        help="racunaj nad celim skupom, bez podele. Sluzi samo "
                             "za proveru: tako dobijene vrednosti ne smeju se "
                             "prijaviti kao rezultat")
    args = parser.parse_args()

    if not os.path.isdir(args.labels):
        raise SystemExit(
            f"Nema mape sa oznakama: {args.labels}\n"
            "Slike oznaciti u YOLO obliku, po pravilima iz odeljka 6.3."
        )

    truth = load_ground_truth(args.images, args.labels, args.min_height)

    if args.no_split:
        print("PAZNJA: racuna se nad celim skupom. Prag se bira i meri nad "
              "istim slikama, pa su vrednosti optimisticne i sluze samo za "
              "proveru.\n")
        tune_truth = eval_truth = truth
    else:
        assignment = dataset_split.get(args.images, dataset_split.FRAMES_CSV,
                                       args.tune_per_session,
                                       rebuild=args.rebuild_split)
        tune_truth = split_truth(truth, assignment, dataset_split.TUNE)
        eval_truth = split_truth(truth, assignment, dataset_split.EVAL)
        if not tune_truth or not eval_truth:
            raise SystemExit("Podela je prazna sa jedne strane. Proveriti "
                             "eval/dataset/split.csv i logs/frames.csv.")

    describe(tune_truth, "Deo za podesavanje")
    total_gt = describe(eval_truth, "Deo za vrednovanje")
    n_images = len(eval_truth)
    print(f"Prag IoU {args.iou}, ciljni odziv {args.target_recall}\n")

    curves, marks, rows = [], [], []

    for name in ("hog", "yolo"):
        path = os.path.join(RESULTS_DIR, f"detections_{name}.csv")
        if not os.path.exists(path):
            print(f"Preskacem {name}: nema {path} (pokrenuti eval/bench.py)")
            continue

        detections = load_detections(path)
        if args.no_split:
            tune_det = eval_det = detections
        else:
            tune_det = split_detections(detections, assignment,
                                        dataset_split.TUNE)
            eval_det = split_detections(detections, assignment,
                                        dataset_split.EVAL)

        # 1. korak: prag se bira na delu za podesavanje
        tune_points, _, _ = evaluate(tune_truth, tune_det, args.iou)
        chosen = operating_point(tune_points, args.target_recall)

        # 2. korak: taj prag se primenjuje na deo za vrednovanje
        eval_points, _, discarded = evaluate(eval_truth, eval_det, args.iou)
        ap = average_precision(eval_points)
        max_recall = max((p[1] for p in eval_points), default=0.0)
        curves.append((name, eval_points, ap))

        print(f"{name}:")
        print(f"  nalaza u delu za vrednovanje {len(eval_det)}, "
              f"AP {ap:.3f}, najveci dostignut odziv {max_recall:.3f}")
        if discarded:
            print(f"  odbaceno na zanemarenim okvirima: {discarded}")

        row = {
            "detector": name,
            "nalaza_vrednovanje": len(eval_det),
            "okvira_vrednovanje": total_gt,
            "slika_vrednovanje": n_images,
            "AP_vrednovanje": round(ap, 4),
            "najveci_odziv_vrednovanje": round(max_recall, 4),
            "ciljni_odziv": args.target_recall,
        }

        if chosen is None:
            print(f"  na delu za podesavanje ne dostize odziv "
                  f"{args.target_recall} ni pri jednom pragu, pa radne tacke "
                  f"nema")
            row.update({k: "" for k in (
                "prag_sa_podesavanja", "odziv_podesavanje", "odziv", "preciznost",
                "odziv_95_od", "odziv_95_do", "preciznost_95_od",
                "preciznost_95_do", "lazno_pozitivnih",
                "lazno_pozitivnih_po_slici")})
            rows.append(row)
            continue

        threshold = chosen[0]
        tp, fp, _, _ = at_threshold(eval_truth, eval_det, args.iou, threshold)
        recall = tp / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fppi = fp / n_images if n_images else 0.0
        r_lo, r_hi = wilson(tp, total_gt)
        p_lo, p_hi = wilson(tp, tp + fp) if (tp + fp) else (None, None)
        marks.append((name, recall, precision))

        print(f"  prag izabran na delu za podesavanje: {threshold:.4f} "
              f"(tamo odziv {chosen[1]:.3f})")
        print(f"  primenjen na deo za vrednovanje: odziv {recall:.3f} "
              f"[{r_lo:.3f}, {r_hi:.3f}], preciznost {precision:.3f}"
              + (f" [{p_lo:.3f}, {p_hi:.3f}]" if p_lo is not None else ""))
        print(f"  lazno pozitivnih {fp}, po slici {fppi:.3f}")

        row.update({
            "prag_sa_podesavanja": round(threshold, 4),
            "odziv_podesavanje": round(chosen[1], 4),
            "odziv": round(recall, 4),
            "preciznost": round(precision, 4),
            "odziv_95_od": round(r_lo, 4),
            "odziv_95_do": round(r_hi, 4),
            "preciznost_95_od": round(p_lo, 4) if p_lo is not None else "",
            "preciznost_95_do": round(p_hi, 4) if p_hi is not None else "",
            "lazno_pozitivnih": fp,
            "lazno_pozitivnih_po_slici": round(fppi, 4),
        })
        rows.append(row)

        curve_path = os.path.join(RESULTS_DIR, f"pr_{name}.csv")
        with open(curve_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prag", "odziv", "preciznost", "lazno_pozitivnih"])
            for score, rec, prec, n_fp in eval_points:
                writer.writerow([f"{score:.6f}", f"{rec:.6f}",
                                 f"{prec:.6f}", n_fp])
        print(f"  kriva -> {curve_path}")

    if not rows:
        raise SystemExit("Nema rezultata. Prvo pokrenuti eval/bench.py.")

    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = os.path.join(RESULTS_DIR, "pr_curve.svg")
    write_pr_svg(curves, svg_path, args.iou, args.target_recall, marks)

    usable = [r for r in rows if r.get("preciznost") != ""]
    if len(usable) == 2:
        a, b = usable
        print("\nPoredjenje pri uporedivom odzivu, na delu za vrednovanje:")
        for r in usable:
            interval = ("" if r["preciznost_95_od"] == "" else
                        f" [{r['preciznost_95_od']:.3f}, "
                        f"{r['preciznost_95_do']:.3f}]")
            print(f"  {r['detector']:5} odziv {r['odziv']:.3f}  "
                  f"preciznost {r['preciznost']:.3f}{interval}")
        if "" not in (a["preciznost_95_od"], a["preciznost_95_do"],
                      b["preciznost_95_od"], b["preciznost_95_do"]):
            preklapaju = not (a["preciznost_95_do"] < b["preciznost_95_od"] or
                              b["preciznost_95_do"] < a["preciznost_95_od"])
            if preklapaju:
                print("  Intervali se PREKLAPAJU: razlika nije veca od merne "
                      "nesigurnosti na ovom broju okvira.")
            else:
                print("  Intervali se ne preklapaju: razlika je veca od merne "
                      "nesigurnosti.")

    print(f"\nSazetak -> {summary_path}")
    print(f"Grafik  -> {svg_path}")


if __name__ == "__main__":
    main()
