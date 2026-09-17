"""Mere vrednovanja detekcije: IoU, preciznost, odziv, AP.

Postupak je onaj iz odeljka 7.6 istrazivackog rada. Nalazi se redjaju po
meri poverenja naniže i uparuju sa tacnim okvirima pri pragu IoU; svaki
tacan okvir moze biti upareni najvise jednom, ostalo su lazno pozitivni.

Kljucni izlaz je poredjenje pri UPOREDIVOM ODZIVU: za zadati ciljni odziv
trazi se prag poverenja koji ga kod svakog postupka daje, pa se tek tada
porede preciznosti. Poredjenje pri podrazumevanim pragovima nije posteno,
jer skorovi dva postupka nisu ista velicina.

Pokretanje:
    python3 eval/metrics.py
    python3 eval/metrics.py --iou 0.5 --target-recall 0.90
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


def evaluate(truth, detections, iou_threshold, ignore_overlap=0.5):
    """Vraca (tacke, broj_okvira). Tacka je (prag, odziv, preciznost,
    lazno_pozitivnih), po opadajucem skoru."""
    total_gt = sum(len(boxes) for boxes, _ in truth.values())
    if total_gt == 0:
        raise SystemExit("Skup nema nijedan oznacen okvir.")

    detections = sorted(detections, key=lambda d: d[5], reverse=True)
    matched = defaultdict(set)

    points = []
    true_positives = 0
    false_positives = 0
    discarded = 0

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
            true_positives += 1
        elif any(intersection_over_detection((x, y, w, h), box)
                 >= ignore_overlap for box in ignored):
            # Nalaz je pao na zanemaren okvir: ne broji se nikako.
            discarded += 1
            continue
        else:
            false_positives += 1

        recall = true_positives / total_gt
        precision = true_positives / (true_positives + false_positives)
        points.append((score, recall, precision, false_positives))

    return points, total_gt, discarded


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

def write_pr_svg(curves, path, iou_threshold, target_recall):
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
    args = parser.parse_args()

    if not os.path.isdir(args.labels):
        raise SystemExit(
            f"Nema mape sa oznakama: {args.labels}\n"
            "Slike oznaciti alatom LabelImg u YOLO obliku."
        )

    truth = load_ground_truth(args.images, args.labels, args.min_height)
    positives = sum(1 for boxes, _ in truth.values() if boxes)
    total_gt = sum(len(boxes) for boxes, _ in truth.values())
    total_ignored = sum(len(ignored) for _, ignored in truth.values())
    n_images = len(truth)

    print(f"Skup: {n_images} slika, od toga {positives} sa osobom, "
          f"{total_gt} oznacenih okvira, {n_images - positives} bez osobe")
    if total_ignored:
        print(f"Zanemarenih okvira: {total_ignored} "
              f"(klasa razlicita od 0 ili visina < {args.min_height:.0f} px)")
    print(f"Prag IoU {args.iou}, ciljni odziv {args.target_recall}\n")

    curves = []
    rows = []

    for name in ("hog", "yolo"):
        path = os.path.join(RESULTS_DIR, f"detections_{name}.csv")
        if not os.path.exists(path):
            print(f"Preskacem {name}: nema {path} (pokrenuti eval/bench.py)")
            continue

        detections = load_detections(path)
        points, _, discarded = evaluate(truth, detections, args.iou)
        ap = average_precision(points)
        best = operating_point(points, args.target_recall)

        curves.append((name, points, ap))

        max_recall = max((p[1] for p in points), default=0.0)
        # Broj lazno pozitivnih po slici u radnoj tacki. Za sistem koji radi
        # neprekidno ta velicina je razumljivija od same preciznosti
        # (odeljak 6.5 rada).
        fppi = best[3] / n_images if best else None

        row = {
            "detector": name,
            "nalaza": len(detections),
            "odbacenih_na_zanemarenim": discarded,
            "AP": round(ap, 4),
            "najveci_odziv": round(max_recall, 4),
            "ciljni_odziv": args.target_recall,
            "prag_za_ciljni_odziv": round(best[0], 4) if best else "",
            "odziv_u_radnoj_tacki": round(best[1], 4) if best else "",
            "preciznost_u_radnoj_tacki": round(best[2], 4) if best else "",
            "lazno_pozitivnih": best[3] if best else "",
            "lazno_pozitivnih_po_slici": round(fppi, 4) if best else "",
        }
        rows.append(row)

        print(f"{name}:")
        print(f"  nalaza {len(detections)}, AP {ap:.3f}, "
              f"najveci dostignut odziv {max_recall:.3f}")
        if discarded:
            print(f"  odbaceno na zanemarenim okvirima: {discarded}")
        if best:
            print(f"  pri odzivu {best[1]:.3f} (prag {best[0]:.3f}) "
                  f"preciznost je {best[2]:.3f}, "
                  f"lazno pozitivnih po slici {fppi:.3f}")
        else:
            print(f"  ne dostize odziv {args.target_recall} ni pri jednom pragu")

        curve_path = os.path.join(RESULTS_DIR, f"pr_{name}.csv")
        with open(curve_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prag", "odziv", "preciznost",
                             "lazno_pozitivnih"])
            for score, recall, precision, fp in points:
                writer.writerow([f"{score:.6f}", f"{recall:.6f}",
                                 f"{precision:.6f}", fp])
        print(f"  kriva -> {curve_path}")

    if not rows:
        raise SystemExit("Nema rezultata. Prvo pokrenuti eval/bench.py.")

    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = os.path.join(RESULTS_DIR, "pr_curve.svg")
    write_pr_svg(curves, svg_path, args.iou, args.target_recall)

    print(f"\nSazetak -> {summary_path}")
    print(f"Grafik  -> {svg_path}")


if __name__ == "__main__":
    main()
