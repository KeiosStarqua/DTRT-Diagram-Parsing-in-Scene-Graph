import argparse
import gzip
import json
import shutil
from pathlib import Path


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def dump_json(data, path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser("Prepare PID2Graph annotations for RelTR")
    parser.add_argument("--src", default="PID2Graph/Complete", help="PID2Graph Complete directory")
    parser.add_argument("--dst", default="data/pid2graph", help="output annotation directory")
    parser.add_argument("--no_gzip_annotations", action="store_true",
                        help="Write train/val/test annotations as plain .json instead of .json.gz")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    rel = {}
    predicates = None
    for split in ["train", "val", "test"]:
        coco_path = src / f"pid_{split}_coco.json"
        rel_path = src / f"pid_{split}_relations.json"
        pred_path = src / f"pid_{split}_predicates.json"

        if not coco_path.exists() or not rel_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"Missing PID2Graph files for split: {split}")

        if args.no_gzip_annotations:
            shutil.copyfile(coco_path, dst / f"{split}.json")
        else:
            gz_path = dst / f"{split}.json.gz"
            with coco_path.open("rb") as src_file, gzip.open(gz_path, "wb", compresslevel=9) as dst_file:
                shutil.copyfileobj(src_file, dst_file)
        rel[split] = load_json(rel_path)
        split_predicates = load_json(pred_path)
        if predicates is None:
            predicates = split_predicates
        elif predicates != split_predicates:
            raise ValueError(f"Predicate list differs for split: {split}")

    rel["rel_categories"] = predicates
    dump_json(rel, dst / "rel.json")

    train_path = dst / ("train.json" if args.no_gzip_annotations else "train.json.gz")
    train_coco = load_json(train_path)
    num_classes = max(cat["id"] for cat in train_coco["categories"]) + 1
    num_rel_classes = len(predicates)
    meta = {
        "num_classes": num_classes,
        "num_rel_classes": num_rel_classes,
        "image_root": str(src),
        "categories": train_coco["categories"],
        "rel_categories": predicates,
    }
    dump_json(meta, dst / "meta.json")

    print(f"Wrote RelTR PID2Graph annotations to {dst}")
    print(f"Use --dataset pid2graph --img_folder {src}/ --ann_path {dst}/")
    print(f"Use --num_classes {num_classes} --num_rel_classes {num_rel_classes}")


if __name__ == "__main__":
    main()
