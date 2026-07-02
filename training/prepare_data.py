"""
prepare_data.py

Run this AFTER you've extracted the Kaggle "Home Fire Dataset" zip into ./data.

It scans ./data for train/val/test folders that each contain an images/
and labels/ subfolder (the standard YOLO layout), sanity-checks image/label
counts, detects which class ids are actually used in the label files, and
writes training/data.yaml for use by train.py.

You said the zip is already split into train/test/val, each with an
images subfolder and a labels subfolder -- this script auto-detects those
folders wherever they land inside ./data (they might be nested one level
deeper depending on how the zip unpacks, e.g. data/home-fire-dataset/train/...).
"""
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_YAML = Path(__file__).resolve().parent / "data.yaml"

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Confirmed class mapping for this dataset (verified against label files):
# class id 0 = fire, class id 1 = smoke
CLASS_NAMES = {0: "fire", 1: "smoke"}


def find_split_dirs():
    """Recursively search data/ for folders matching train/val/test that
    each contain an images/ and labels/ subfolder."""
    found = {}
    for split, aliases in SPLIT_ALIASES.items():
        for path in DATA_ROOT.rglob("*"):
            if not path.is_dir() or path.name.lower() not in aliases:
                continue
            images_dir = None
            labels_dir = None
            for sub in path.iterdir():
                if sub.is_dir() and sub.name.lower() in ("images", "image", "imgs"):
                    images_dir = sub
                if sub.is_dir() and sub.name.lower() in ("labels", "label", "annotations"):
                    labels_dir = sub
            if images_dir and labels_dir:
                found[split] = {"images": images_dir, "labels": labels_dir}
                break  # take the first valid match for this split
    return found


def detect_classes(splits):
    class_ids = set()
    for info in splits.values():
        for txt_file in info["labels"].rglob("*.txt"):
            with open(txt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_ids.add(int(line.split()[0]))
    return sorted(class_ids)


def sanity_check(splits):
    for split, info in splits.items():
        images = [p for p in info["images"].rglob("*") if p.suffix.lower() in IMG_EXTS]
        labels = list(info["labels"].rglob("*.txt"))
        print(
            f"[{split}] images={len(images)}  labels={len(labels)}\n"
            f"    images_dir = {info['images']}\n"
            f"    labels_dir = {info['labels']}"
        )
        if len(images) == 0:
            print(f"    WARNING: no images found for split '{split}'")
        if len(labels) == 0:
            print(f"    WARNING: no labels found for split '{split}'")


def main():
    if not DATA_ROOT.exists():
        sys.exit(f"ERROR: {DATA_ROOT} does not exist. Extract the dataset zip into ./data first.")

    splits = find_split_dirs()

    if "train" not in splits:
        sys.exit(
            "ERROR: could not auto-detect a 'train' split with images/ and labels/ "
            "subfolders anywhere inside ./data. Print the folder tree (e.g. "
            "`find data -maxdepth 4 -type d`) and check the actual layout -- "
            "the zip may have unpacked into an extra nested folder."
        )

    if "val" not in splits:
        print("WARNING: no 'val'/'valid' split found.")
        if "test" in splits:
            print("Using 'test' split as validation data too.")
            splits["val"] = splits["test"]
        else:
            sys.exit("ERROR: no val or test split found -- cannot train without validation data.")

    sanity_check(splits)

    class_ids = detect_classes(splits)
    print(f"\nDetected class ids in label files: {class_ids}")

    unexpected = [c for c in class_ids if c not in CLASS_NAMES]
    if unexpected:
        sys.exit(
            f"ERROR: found unexpected class id(s) {unexpected} in the label files, "
            f"but only {list(CLASS_NAMES.keys())} (fire, smoke) were expected. "
            "The dataset labels don't match the confirmed mapping -- check the "
            "label files before continuing."
        )

    print(f"Using confirmed mapping -> {CLASS_NAMES}")

    data_yaml = {
        "path": str(DATA_ROOT.resolve()),
        "train": str(splits["train"]["images"].resolve()),
        "val": str(splits["val"]["images"].resolve()),
        "names": {i: CLASS_NAMES[i] for i in sorted(CLASS_NAMES)},
    }
    if "test" in splits:
        data_yaml["test"] = str(splits["test"]["images"].resolve())

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nWrote {OUTPUT_YAML}")
    print("Review it, then run: python training/train.py")


if __name__ == "__main__":
    main()
