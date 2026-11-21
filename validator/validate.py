"""Validate a Day 10 delivery and generate candidates for human review."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pngutil import read_rgb


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def classes(root):
    return {int(i): name for i, name in
            (line.split(maxsplit=1) for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines())}


def boxes(path, width, height):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, x, y, w, h = line.split()
            x, y, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
            yield int(i), (x-w/2, y-h/2, x+w/2, y+h/2)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_counts(path, rects):
    width, height, rgb = read_rgb(path)
    totals = Counter()
    hits = [Counter() for _ in rects]
    rows = defaultdict(list)
    for i, (x1, y1, x2, y2) in enumerate(rects):
        for y in range(max(0, round(y1)), min(height, round(y2))):
            rows[y].append((i, max(0, round(x1)), min(width, round(x2))))
    for y in range(height):
        ids = []
        for x in range(width):
            offset = 3 * (y * width + x)
            ids.append(rgb[offset] + 256 * rgb[offset + 1] + 65536 * rgb[offset + 2])
        totals.update(ids)
        for i, x1, x2 in rows[y]:
            hits[i].update(ids[x1:x2])
    return totals, hits


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0


def review_candidates(target, det, segments, cat, names, totals, hits):
    for seg in segments:
        sid = seg["id"]
        x, y, w, h = seg["bbox"]
        rect = (x, y, x+w, y+h)
        label = cat[seg["category_id"]]["name"]
        area = totals[sid]
        if area < 64 or seg.get("iscrowd") or not cat[seg["category_id"]].get("isthing"):
            continue
        matches = [(i, iou(rect, box), hits[i][sid] / area)
                   for i, (_, box) in enumerate(det)]
        same = [m for m in matches if names[det[m[0]][0]] == label]
        same.sort(key=lambda m: m[1], reverse=True)
        covered = [m for m in same if m[2] >= .22]
        best = max(matches, key=lambda m: m[1], default=(None, 0, 0))
        kind = None
        indices = []
        if len(covered) >= 2 and all(m[2] <= .79 for m in covered[:2]):
            kind, indices = "instance_merge", [m[0] for m in covered[:2]]
        elif same and .15 <= same[0][1] and .15 <= same[0][2] < .83:
            kind, indices = "bbox_drift", [same[0][0]]
        elif best[1] >= .60 and best[2] >= .75 and names[det[best[0]][0]] != label and not any(m[1] >= .30 for m in same):
            kind, indices = "class_swap_or_category_swap", [best[0]]
        elif area >= 500 and best[1] < .15 and all(m[2] < .25 for m in matches):
            kind = "missing_box"
        if kind:
            yield dict(target=target, type=kind, segment=sid, panoptic_class=label,
                       panoptic_bbox=rect,
                       detection=[dict(label=names[det[i][0]], bbox=det[i][1]) for i in indices])


def validate(root, with_candidates):
    manifest = read_json(root / "manifest.json")["items"]
    pan = read_json(root / "panoptic.json")
    dimensions = {im["id"]: im for im in pan["images"]}
    categories = {cat["id"]: cat for cat in pan["categories"]}
    names = classes(root)
    jobs = {row["job_id"]: row for row in csv_rows(root / "jobs.csv")}
    scenes = defaultdict(set)
    errors = set()
    candidates = []
    for row in read_json(root / "metadata.json")["items"]:
        scenes[row["scene_id"]].add(row["split"])
        if row["guideline_version"] != jobs[row["job_id"]]["guideline"]:
            errors.add(("guideline_mismatch", row["image"]))
    errors.update(("scene_leak", scene) for scene, splits in scenes.items() if len(splits) > 1)
    for ann in pan["annotations"]:
        image = dimensions[ann["image_id"]]
        target = Path(image["file_name"]).stem
        entry = manifest[target]
        image_path = root / entry["image"]
        rules = set()
        if not image_path.is_file():
            rules.add("missing_image")
        if any((root / entry[key]).is_file() and sha(root / entry[key]) != entry[key + "_sha256"]
               for key in ("image", "label", "mask")):
            rules.add("wrong_hash")
        width, height = image["width"], image["height"]
        det = list(boxes(root / entry["label"], width, height))
        if any(i not in names for i, _ in det):
            rules.add("invalid_class")
        if any(x1 < -.001 or y1 < -.001 or x2 > width+.001 or y2 > height+.001 or
               x1 >= x2 or y1 >= y2 for _, (x1, y1, x2, y2) in det):
            rules.add("bbox_out_of_bounds")
        for (i, a), (j, b) in combinations(det, 2):
            if i == j and all(abs((a[k+2]-a[k])-(b[k+2]-b[k])) < .01 for k in (0, 1)) and max(abs(a[k]-b[k]) for k in (0, 1)) <= 2.01:
                rules.add("duplicate_box")
                break
        segments = ann["segments_info"]
        totals, hits = mask_counts(root / entry["mask"], [b for _, b in det] if with_candidates and not rules else [])
        if any(s["category_id"] not in categories for s in segments):
            rules.add("invalid_category_id")
        if any(categories.get(s["category_id"], {}).get("isthing") and s["area"] > totals[s["id"]] for s in segments):
            rules.add("thing_overlap")
        if with_candidates and not rules:
            candidates.extend(review_candidates(target, det, segments, categories, names, totals, hits))
        errors.update((kind, target) for kind in rules)
    return {"errors": [dict(type=kind, target=target) for kind, target in sorted(errors)]}, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    errors, candidates = validate(args.delivery, args.candidates is not None)
    write_json(args.out, errors)
    if args.candidates:
        write_json(args.candidates, {"candidates": candidates})
    print(f"{len(errors['errors'])} rule findings, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
