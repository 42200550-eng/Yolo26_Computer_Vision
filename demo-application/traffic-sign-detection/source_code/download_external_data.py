import argparse
import csv
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCES = [
    # Example source object:
    # {
    #   "name": "gtsdb",
    #   "url": "https://.../gtsdb.zip",
    #   "type": "zip",
    #   "labels_format": "csv",
    #   "annotations_csv": "gt.csv",
    #   "class_map": {"17": 0, "1": 1}
    # }
]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(target))


def unzip_if_needed(path: Path, dst: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dst)


def find_images(root: Path) -> list[Path]:
    out = []
    for ext in IMAGE_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    return sorted(out)


def parse_sidecar_yolo(label_path: Path, class_map: dict[int, int]) -> list[str]:
    lines_out = []
    if not label_path.exists():
        return lines_out

    for ln in label_path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        try:
            src_cls = int(parts[0])
            dst_cls = class_map.get(src_cls, src_cls)
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        lines_out.append(f"{dst_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines_out


def parse_csv_annotations(
    csv_path: Path,
    class_map: dict[int, int],
    filename_key: str,
    class_key: str,
    xmin_key: str,
    ymin_key: str,
    xmax_key: str,
    ymax_key: str,
    width_key: str,
    height_key: str,
) -> dict[str, list[str]]:
    by_file = defaultdict(list)
    if not csv_path.exists():
        return by_file

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                src_cls = int(row[class_key])
                if src_cls not in class_map:
                    continue
                dst_cls = class_map[src_cls]
                x1 = float(row[xmin_key])
                y1 = float(row[ymin_key])
                x2 = float(row[xmax_key])
                y2 = float(row[ymax_key])
                width = float(row[width_key]) if width_key in row and row[width_key] else (x2 - x1)
                height = float(row[height_key]) if height_key in row and row[height_key] else (y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                xc = (x1 + x2) / 2.0 / width
                yc = (y1 + y2) / 2.0 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    continue
                by_file[row[filename_key]].append(f"{dst_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            except (KeyError, ValueError):
                continue
    return by_file


def copy_images_with_labels(
    src_root: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    labels_format: str,
    class_map: dict[int, int],
    annotations_csv: str,
    csv_schema: dict,
) -> tuple[int, int]:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(src_root)
    copied_images = 0
    copied_labels = 0

    csv_label_map = {}
    if labels_format == "csv":
        csv_path = src_root / annotations_csv
        csv_label_map = parse_csv_annotations(
            csv_path=csv_path,
            class_map=class_map,
            filename_key=csv_schema.get("filename", "filename"),
            class_key=csv_schema.get("class_id", "class_id"),
            xmin_key=csv_schema.get("xmin", "xmin"),
            ymin_key=csv_schema.get("ymin", "ymin"),
            xmax_key=csv_schema.get("xmax", "xmax"),
            ymax_key=csv_schema.get("ymax", "ymax"),
            width_key=csv_schema.get("width", "width"),
            height_key=csv_schema.get("height", "height"),
        )

    for p in images:
        lines = []
        if labels_format == "yolo":
            lines = parse_sidecar_yolo(p.with_suffix(".txt"), class_map)
        elif labels_format == "csv":
            lines = csv_label_map.get(p.name, [])

        if not lines:
            continue

        out_img = dst_img_dir / p.name
        out_lbl = dst_lbl_dir / f"{p.stem}.txt"
        out_img.write_bytes(p.read_bytes())
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


def save_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["name", "status", "note"])
            writer.writeheader()
            writer.writerows(rows)
        return
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare external traffic-sign images")
    parser.add_argument("--output-root", default="external_data", help="Output root folder")
    parser.add_argument("--sources-json", default="", help="JSON file of source list [{name,url,type}]")
    parser.add_argument("--manifest", default="external_data/manifest.json", help="Download manifest output")
    parser.add_argument("--dataset-out", default="datasets_v2", help="Destination dataset root for train split")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    raw_root = output_root / "raw"
    images_root = output_root / "images"
    dataset_out = Path(args.dataset_out).resolve()
    dataset_train_img = dataset_out / "images" / "train"
    dataset_train_lbl = dataset_out / "labels" / "train"

    if args.sources_json:
        sources = json.loads(Path(args.sources_json).read_text(encoding="utf-8"))
    else:
        sources = DEFAULT_SOURCES

    manifest_rows = []

    for src in sources:
        name = src.get("name", "unknown")
        url = src.get("url", "")

        if not url:
            manifest_rows.append({"name": name, "status": "skipped", "note": "missing_url"})
            continue

        archive_path = raw_root / name / Path(url).name
        extract_root = raw_root / name / "extracted"

        try:
            print(f"[INFO] Downloading {name}: {url}")
            download_file(url, archive_path)
            unzip_if_needed(archive_path, extract_root)
            src_root = extract_root if extract_root.exists() else archive_path.parent
            labels_format = str(src.get("labels_format", "yolo")).lower()
            class_map_raw = src.get("class_map", {})
            class_map = {int(k): int(v) for k, v in class_map_raw.items()}
            annotations_csv = str(src.get("annotations_csv", ""))
            csv_schema = src.get("csv_schema", {})

            copied_images, copied_labels = copy_images_with_labels(
                src_root=src_root,
                dst_img_dir=dataset_train_img,
                dst_lbl_dir=dataset_train_lbl,
                labels_format=labels_format,
                class_map=class_map,
                annotations_csv=annotations_csv,
                csv_schema=csv_schema,
            )

            copied_to_raw = len(find_images(src_root))
            images_root.mkdir(parents=True, exist_ok=True)
            for p in find_images(src_root):
                out = images_root / p.name
                if not out.exists():
                    out.write_bytes(p.read_bytes())

            manifest_rows.append(
                {
                    "name": name,
                    "status": "ok",
                    "note": f"raw_images={copied_to_raw},train_images={copied_images},train_labels={copied_labels}",
                }
            )
        except Exception as e:
            manifest_rows.append({"name": name, "status": "error", "note": str(e)})

    save_manifest(Path(args.manifest).resolve(), manifest_rows)

    print("[INFO] External data preparation completed")
    print(f"[INFO] Images directory: {images_root}")
    print(f"[INFO] Dataset train images: {dataset_train_img}")
    print(f"[INFO] Dataset train labels: {dataset_train_lbl}")


if __name__ == "__main__":
    main()
