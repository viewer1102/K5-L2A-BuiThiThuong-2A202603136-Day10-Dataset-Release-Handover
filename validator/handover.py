"""Prepare and finish a Day 10 run after manual review."""
import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from validate import boxes, classes, csv_rows, read_json, sha, validate, write_json

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = {"bbox_drift", "missing_box", "class_swap", "category_swap", "instance_merge"}
PANOPTIC = {"invalid_category_id", "thing_overlap", "category_swap", "instance_merge"}
REWORK = {"missing_image", "invalid_class", "bbox_out_of_bounds", "duplicate_box",
          "class_swap", "bbox_drift", "missing_box"}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def inputs(delivery):
    manifest = read_json(delivery / "manifest.json")["items"]
    meta = {r["image"]: r for r in read_json(delivery / "metadata.json")["items"]}
    jobs = {r["job_id"]: r for r in csv_rows(delivery / "jobs.csv")}
    return manifest, meta, jobs


def trace(delivery, output, manifest, meta, jobs):
    change = (delivery / "guideline/changelog.md").read_text(encoding="utf-8")
    ticket = (delivery / "incoming/ticket.md").read_text(encoding="utf-8")
    label = re.search(r"Lớp `([^`]+)`", change).group(1)
    previous = re.search(r"\((v[\d.]+) cho phép", change).group(1)
    scenes = set(re.findall(r"scene_\d+", ticket))
    class_id = next(k for k, v in classes(delivery).items() if v == label)
    a, b, people = [], [], set()
    for image, row in sorted(meta.items()):
        job = jobs[row["job_id"]]
        labels = delivery / manifest[image]["label"]
        if job["guideline"] == previous and any(i == class_id for i, _ in boxes(labels, 1, 1)):
            (a if row["scene_id"] in scenes else b).append(image)
            people.update((job["annotator"], job["reviewer"]))
    write_json(output / "trace.json", dict(tier_a=a, tier_b=b, notify=sorted(people)))


def prepare(delivery, output):
    manifest, meta, jobs = inputs(delivery)
    errors, candidates = validate(delivery, True)
    write_json(output / "rules.json", errors)
    write_json(output / "candidates.json", {"candidates": candidates})
    trace(delivery, output, manifest, meta, jobs)
    if not (output / "review_log.csv").exists():
        pending = sorted({(c["target"], c["type"]) for c in candidates})
        write_csv(output / "review_log.csv", ["image", "type", "verdict", "note"],
                  [(image, kind, "", "") for image, kind in pending])
    if not (output / "hash_review.csv").exists():
        write_csv(output / "hash_review.csv", ["image", "verdict"],
                  [(e["target"], "") for e in errors["errors"] if e["type"] == "wrong_hash"])
    page = ["<!doctype html><meta charset=utf-8><title>Day 10 review</title>",
            "<h1>Day 10 — Review</h1><p>Cam: bbox panoptic. Xanh: bbox YOLO.</p>"]
    sizes = {Path(im["file_name"]).stem: im for im in read_json(delivery / "panoptic.json")["images"]}
    for finding in candidates + [e for e in errors["errors"] if e["type"] == "wrong_hash"]:
        name = finding["target"]
        im = sizes[name]
        w, h = im["width"], im["height"]
        uri = Path(os.path.relpath(delivery / manifest[name]["image"], output)).as_posix()
        page.append(f'<h2>{html.escape(name)} — {html.escape(finding["type"])}</h2>')
        page.append(f'<p>{html.escape(str(finding))}</p><svg viewBox="0 0 {w} {h}" width="{min(800,w*2)}">')
        page.append(f'<image href="{uri}" width="{w}" height="{h}"/>')
        outlines = [("orange", finding["panoptic_bbox"])] if "panoptic_bbox" in finding else []
        outlines += [("lime", b["bbox"]) for b in finding.get("detection", [])]
        for color, (x, y, x2, y2) in outlines:
            page.append(f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" fill="none" stroke="{color}" stroke-width="2"/>')
        page.append("</svg>")
    (output / "review.html").write_text("\n".join(page), encoding="utf-8")
    print(f"{len(errors['errors'])} lỗi luật, {len(candidates)} ứng viên. Mở {output / 'review.html'}.")


def run_script(name, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], cwd=ROOT, check=True)


def finish(run, face):
    delivery = run / "delivery"
    manifest, meta, jobs = inputs(delivery)
    errors = {(e["type"], e["target"]) for e in read_json(run / "rules.json")["errors"]}
    review = csv_rows(run / "review_log.csv")
    reviewed = set()
    for row in review:
        image, kind, verdict, note = (row[k].strip() for k in ("image", "type", "verdict", "note"))
        if image not in manifest or verdict not in {"confirmed", "rejected"} or len(note) < 20:
            raise SystemExit(f"Review chưa hợp lệ: {row}")
        reviewed.add((image, kind))
        if verdict == "confirmed":
            if kind not in SEMANTIC:
                raise SystemExit(f"Loại lỗi chưa hợp lệ: {kind}")
            errors.add((kind, image))
    for candidate in read_json(run / "candidates.json")["candidates"]:
        image, kind = candidate["target"], candidate["type"]
        kinds = {kind, "class_swap", "category_swap"} if kind == "class_swap_or_category_swap" else {kind}
        if not any((image, k) in reviewed for k in kinds):
            raise SystemExit(f"Còn ứng viên chưa xem: {image} {kind}")
    hash_review = {row["image"]: row["verdict"].strip() for row in csv_rows(run / "hash_review.csv")}
    for kind, image in errors:
        if kind == "wrong_hash" and hash_review.get(image) not in {"intact", "damaged"}:
            raise SystemExit(f"Cần mở ảnh sai hash và điền intact/damaged: {image}")
    if face not in manifest or not (delivery / manifest[face]["image"]).is_file():
        raise SystemExit("Cần tên một ảnh đã xem có mặt người rõ, không kèm đuôi.")
    write_json(run / "errors.json", {"errors": [dict(type=k, target=i) for k, i in sorted(errors)]})
    release = read_json(delivery / "release.json")["release_id"]
    v2 = run / "handover" / (release + "_v2")
    if v2.exists():
        raise SystemExit(f"Đã có bản bàn giao ở {v2}. Dời bản đó sang thư mục lưu riêng trước khi tạo lại.")
    names = classes(delivery)
    split_counts = defaultdict(Counter)
    for row in meta.values():
        split_counts[row["scene_id"]][row["split"]] += 1
    by_image = defaultdict(set)
    for kind, image in errors:
        by_image[image].add(kind)
    for folder in ("images", "labels"):
        (v2 / folder).mkdir(parents=True)
    entries, rework, tracking, pan_notes = [], [], [], []
    counts = Counter()
    for image, entry in sorted(manifest.items()):
        row = meta[image]
        job = jobs[row["job_id"]]
        assigned = (job["annotator"], job["reviewer"])
        kinds = by_image[image]
        fixes = kinds & REWORK
        if "wrong_hash" in kinds and hash_review[image] == "damaged":
            fixes.add("wrong_hash")
        if fixes:
            for kind in sorted(fixes):
                rework.append((image, kind, *assigned, "Khôi phục ảnh hoặc gán lại và kiểm lại"))
            continue
        if kinds & PANOPTIC:
            pan_notes.append(image + " (" + ", ".join(sorted(kinds & PANOPTIC)) + ")")
        for key in ("image", "label"):
            rel = entry[key]
            shutil.copyfile(delivery / rel, v2 / rel)
            entries.append((rel, sha(v2 / rel)))
        counts.update(names[i] for i, _ in boxes(v2 / entry["label"], 1, 1))
        scene = row["scene_id"]
        split = split_counts[scene].most_common(1)[0][0]
        tracking.append((image, scene, split, row["job_id"], *assigned, job["guideline"]))
    write_csv(v2 / "manifest.csv", ["file", "sha256"], sorted(entries))
    write_csv(v2 / "label_summary.csv", ["class", "count"], [*sorted(counts.items()), ("TOTAL", sum(counts.values()))])
    write_csv(v2 / "truyvet.csv", ["image", "scene_id", "split", "job_id", "annotator", "reviewer", "guideline"], tracking)
    write_csv(v2 / "rework_queue.csv", ["image", "type", "annotator", "reviewer", "action"], rework)
    affected = len({image for _, image in errors})
    rate = affected / len(manifest)
    decision = "Hold" if rate > .02 else "Release"
    guidelines = ", ".join(sorted({job["guideline"] for job in jobs.values()}))
    notes = "; ".join(pan_notes) or "không có lỗi panoptic đã xác nhận"
    (v2 / "README.txt").write_text(
        f"Release: {v2.name}\nParent: {release}\nSố ảnh: {len(tracking)}\nSố nhãn: {sum(counts.values())}\n"
        f"Guideline: {guidelines} theo jobs.csv\nQC v1: {affected}/{len(manifest)} = {rate:.2%}\n"
        f"Kết luận v1: {decision} theo ngưỡng 2%\n"
        f"Kết luận v2: Release cho detection sau khi tách {len({r[0] for r in rework})} ảnh vào rework\n"
        f"Ghi chú: Split theo đa số của cảnh. Guideline theo job. SHA-256 tính lại sau review ảnh sai hash. "
        f"Người xử lý xem rework_queue.csv. Báo team panoptic: {notes}.\n", encoding="utf-8")
    (run / "security.md").write_text(
        "# Bảo mật khi làm việc\n\n"
        "## 1. Ca USB\nGiữ dữ liệu trong môi trường được cấp quyền. Xin người phụ trách cấp phiên làm từ xa "
        "hoặc bố trí thời gian trên máy công ty. Email cá nhân và nhờ người ngoài xử lý đều đưa dữ liệu ra ngoài phạm vi cấp quyền.\n\n"
        "## 2. Năm việc\nĐược mở job được giao trên máy công ty theo quyền. Không gửi ảnh màn hình cho người quen. "
        "Không chép lô vào USB cá nhân. Không dán nhãn dự án thật vào chatbot. Không chia sẻ tài khoản CVAT, "
        "đề nghị cấp tài khoản riêng nếu đồng nghiệp cần quyền.\n\n"
        f"## 3. Ca khuôn mặt\nẢnh {face} là ví dụ em đã kiểm tra trực tiếp. Đặt lô ở trạng thái Hold và "
        "báo người phụ trách dữ liệu hoặc an toàn thông tin. Ghi tên ảnh và bộ guideline áp dụng trong kênh nội bộ. "
        "Việc làm mờ hoặc loại ảnh phải qua quy trình phê duyệt để giữ nguyên chứng cứ và truy vết được bản sửa.\n",
        encoding="utf-8")
    lineage = run.parent / "lineage.jsonl"
    previous = lineage.read_text(encoding="utf-8").splitlines() if lineage.exists() else []
    events = [json.loads(line) for line in previous if line.strip()]
    if not any(e.get("event") == "decision" and e.get("release_id") == release and e.get("decision") == decision.upper() for e in events):
        run_script("lineage.py", "record", "--event", "decision", "--release-id", release, "--decision", decision.upper())
    run_script("lineage.py", "impact", "--release-id", release)
    run_script("pack_release.py", v2)
    if not any(e.get("event") == "release" and e.get("release_id") == v2.name and e.get("parent") == release for e in events):
        run_script("lineage.py", "record", "--event", "release", "--release-id", v2.name, "--parent", release)
    run_script("check_handover.py", "--run", run)
    run_script("export_submission.py", "--run", run)
    run_script("score_findings.py", "--run", run)
    scoring = (run / "report.md").read_text(encoding="utf-8")
    candidate_images = len({c["target"] for c in read_json(run / "candidates.json")["candidates"]})
    calib = run.parent / "clean" / "candidates.json"
    clean_count = str(len(read_json(calib)["candidates"])) if calib.is_file() else "chưa đo"
    (ROOT / "submission/REPORT.md").write_text(
        "# Day 10 — Báo cáo cá nhân\n\n" + scoring +
        f"\n## Hàng đợi review\n- Ảnh cần xem (seed luyện tập): {candidate_images}.\n"
        f"- Số ứng viên lô sạch 50 ảnh: {clean_count}.\n"
        "- Ngưỡng khởi tạo theo IoU, phần pixel mask nằm trong box và diện tích segment. "
        "Các ca lệch tự nhiên được ghi rejected trong review_log.csv. Nếu đưa cả các ca đó vào errors.json, "
        "false positive tăng, precision giảm và kéo F1 xuống.\n\n"
        f"## Quyết định và lineage\nQC v1: {affected}/{len(manifest)} = {rate:.2%}, kết luận {decision}. "
        "R-021 đã dùng v1, nên cần đánh giá lại hoặc train lại bằng bản phù hợp.\n\n"
        "## SHA-256\nHash giống nhau cho thấy file khớp về byte với bản tham chiếu. Muốn biết box và lớp đúng "
        "cần xem ảnh và áp dụng guideline.\n\n"
        "## Truy vết\nTầng A là ảnh trong cảnh bị báo, thuộc job dùng guideline cũ và có box của lớp đổi quy tắc. "
        "Tầng B là các ảnh cùng điều kiện ở cảnh khác. Dùng phiên bản theo job để tránh metadata ảnh bị sai.\n",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "finish"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--face")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.phase == "prepare":
        prepare(run / "delivery", run)
    else:
        finish(run, args.face)
