"""
Create a validation subset ("val") by moving ~15% of *base frames* from "train".
For each selected base frame, move BOTH image modalities:
  - <name>_vis.png
  - <name>_nir.png
and move ALL THREE corresponding labels for that frame:
  - <name>_color.png
  - <name>_labelids.png
  - <name>_instanceids.png

The script preserves the directory structure:
  .../images/train/<sequence>/** -> .../images/val/<sequence>/**
  .../labels/train/<sequence>/** -> .../labels/val/<sequence>/**

Supported filename conventions:
- Suffix patterns: *_vis.png / *_nir.png -> labels *_color.png, *_labelids.png, *_instanceids.png
- Special tokens: "windshield_vis"/"windshield_nir" -> labels with "color"/"labelids"/"instanceids"
- Arbitrary nesting inside each <sequence> folder (uses os.walk).

Only CLI option: --dry-run (preview actions, no files moved).

Usage:
  Preview only:
      python split_dataset.py --dry-run
  Execute:
      python split_dataset.py
"""

import os
import shutil
import random
import argparse
from typing import Dict, List, Optional, Tuple

ROOT = "/home/lrm/workspace/GOOSE"

# Fixed sampling behavior
FRACTION = 0.15      # 15% per sequence (by unique base frames)
SEED = 42            # reproducible selection
OVERWRITE = False    # do not overwrite if destination already exists


IMAGES_TRAIN = os.path.join(ROOT, "images", "train")
LABELS_TRAIN = os.path.join(ROOT, "labels", "train")
IMAGES_VAL   = os.path.join(ROOT, "images", "val")
LABELS_VAL   = os.path.join(ROOT, "labels", "val")

LABEL_TYPES = ("color", "labelids", "instanceids")

def ensure_dirs() -> None:
    """Ensure destination directories exist."""
    os.makedirs(IMAGES_VAL, exist_ok=True)
    os.makedirs(LABELS_VAL, exist_ok=True)


def list_sequences(images_train_root: str) -> List[str]:
    """List immediate sub-directories (sequences) under images/train."""
    if not os.path.isdir(images_train_root):
        raise FileNotFoundError(f"images/train not found: {images_train_root}")
    seqs = []
    for name in sorted(os.listdir(images_train_root)):
        full = os.path.join(images_train_root, name)
        if os.path.isdir(full):
            seqs.append(name)
    return seqs


def detect_modality_and_base(stem: str) -> Tuple[Optional[str], str]:
    """
    Given filename *without extension*, detect modality ('vis'|'nir') and
    derive a normalized base name shared by both modalities.

    Supported patterns:
      ..._vis        -> modality='vis', base without the trailing '_vis'
      ..._nir        -> modality='nir', base without the trailing '_nir'
      ...windshield_vis -> modality='vis', base with 'windshield_vis' -> 'windshield'
      ...windshield_nir -> modality='nir', base with 'windshield_nir' -> 'windshield'
    """
    base = stem
    modality: Optional[str] = None

    if "windshield_vis" in base:
        modality = "vis"
        base = base.replace("windshield_vis", "windshield")
    elif "windshield_nir" in base:
        modality = "nir"
        base = base.replace("windshield_nir", "windshield")

    if base.endswith("_vis"):
        modality = "vis"
        base = base[: -len("_vis")]
    elif base.endswith("_nir"):
        modality = "nir"
        base = base[: -len("_nir")]

    return modality, base


def walk_images_collect_bases(seq_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Recursively walk a sequence directory and collect base frames.
    Returns a dict: base_key -> {'vis': path or None, 'nir': path or None}
    where base_key is the relative path (dir + normalized base name).
    """
    bases: Dict[str, Dict[str, Optional[str]]] = {}
    for root, _, files in os.walk(seq_dir):
        for fn in files:
            if not fn.endswith(".png"):
                continue
            stem, _ = os.path.splitext(fn)
            modality, base_name = detect_modality_and_base(stem)
            if modality not in ("vis", "nir"):
                continue  # skip non-vis/nir images
            rel_dir = os.path.relpath(root, IMAGES_TRAIN)  # sequence/... relative to IMAGES_TRAIN
            base_key = os.path.join(rel_dir, base_name)

            entry = bases.setdefault(base_key, {"vis": None, "nir": None})
            full_path = os.path.join(root, fn)
            entry[modality] = full_path
    return bases


def _pick_label_path_for_type(dirn: str, fname: str, label_type: str) -> Optional[str]:
    """
    Given a mirrored labels directory (dirn) and an image filename (fname),
    propose a path for a single label_type and return the first that exists.
    Order of attempts:
      1) windshield_vis/nir -> label_type
      2) _vis/_nir suffix   -> _{label_type}
      3) fallback token replace ('vis'/'nir' -> label_type)
    """
    if "windshield_vis" in fname:
        cand = os.path.join(dirn, fname.replace("windshield_vis", label_type))
        if os.path.exists(cand):
            return cand
    if "windshield_nir" in fname:
        cand = os.path.join(dirn, fname.replace("windshield_nir", label_type))
        if os.path.exists(cand):
            return cand

    if fname.endswith("_vis.png"):
        cand = os.path.join(dirn, fname.replace("_vis.png", f"_{label_type}.png"))
        if os.path.exists(cand):
            return cand
    if fname.endswith("_nir.png"):
        cand = os.path.join(dirn, fname.replace("_nir.png", f"_{label_type}.png"))
        if os.path.exists(cand):
            return cand

    stem, ext = os.path.splitext(fname)
    if "vis" in stem:
        cand = os.path.join(dirn, stem.replace("vis", label_type) + ext)
        if os.path.exists(cand):
            return cand
    if "nir" in stem:
        cand = os.path.join(dirn, stem.replace("nir", label_type) + ext)
        if os.path.exists(cand):
            return cand

    return None


def label_triplet_for_image(img_path: str) -> Dict[str, str]:
    """
    From an image path inside /images/train/<seq>/..., produce a dict with
    existing paths for all three label types: {'color':..., 'labelids':..., 'instanceids':...}.
    Returns {} if any of the three is missing.
    """
    rel_img = os.path.relpath(img_path, IMAGES_TRAIN)
    dirn = os.path.join(LABELS_TRAIN, os.path.dirname(rel_img))
    fname = os.path.basename(rel_img)

    found: Dict[str, str] = {}
    for lt in LABEL_TYPES:
        p = _pick_label_path_for_type(dirn, fname, lt)
        if p is None:
            return {}  # require all three to exist
        found[lt] = p
    return found


def move_file(src: str, dst: str, dry_run: bool) -> str:
    """Move a file, creating parent dirs; optionally dry-run. Returns status string."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        if OVERWRITE:
            if not dry_run:
                os.remove(dst)
        else:
            return "skipped (exists)"
    if not dry_run:
        shutil.move(src, dst)
    return "moved" if not dry_run else "would-move"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move ~15% of base frames from train to val. For each base, move VIS+NIR images and ALL THREE labels (color, labelids, instanceids), preserving folder structure.",
        add_help=True
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without moving files.")
    args = parser.parse_args()

    rng = random.Random(SEED)
    ensure_dirs()
    sequences = list_sequences(IMAGES_TRAIN)

    overall = {
        "total_bases_with_vis_nir": 0,
        "total_eligible_with_3_labels": 0,
        "total_planned_bases": 0,
        "total_images_moved": 0,
        "total_labels_moved": 0,
        "sequences": {}
    }

    for seq in sequences:
        seq_dir = os.path.join(IMAGES_TRAIN, seq)
        bases = walk_images_collect_bases(seq_dir)

        bases_with_both = {k: v for k, v in bases.items() if v.get("vis") and v.get("nir")}
        n_with_both = len(bases_with_both)
        overall["total_bases_with_vis_nir"] += n_with_both

        eligible = []
        labels_cache: Dict[str, Dict[str, str]] = {}
        for base_key, entry in bases_with_both.items():
            vis_path = entry["vis"]
            triplet = label_triplet_for_image(vis_path)
            if triplet:
                eligible.append(base_key)
                labels_cache[base_key] = triplet

        n_eligible = len(eligible)
        overall["total_eligible_with_3_labels"] += n_eligible

        if n_eligible == 0:
            overall["sequences"][seq] = {
                "bases_with_vis_nir": n_with_both,
                "eligible_with_3_labels": 0,
                "planned_bases": 0,
                "images_moved": 0,
                "labels_moved": 0,
                "note": "No base with VIS+NIR and all 3 labels"
            }
            continue

        # Select ~15% of eligible bases (at least 1)
        target = max(1, int(round(n_eligible * FRACTION)))
        rng.shuffle(eligible)
        selected = eligible[:target]

        imgs_moved = 0
        labs_moved = 0

        for base_key in selected:
            entry = bases_with_both[base_key]
            for modality in ("vis", "nir"):
                img_src = entry[modality]
                rel_img = os.path.relpath(img_src, IMAGES_TRAIN)
                img_dst = os.path.join(IMAGES_VAL, rel_img)
                status_img = move_file(img_src, img_dst, args.dry_run)
                if "moved" in status_img:
                    imgs_moved += 1

            triplet = labels_cache[base_key]  # already validated
            for lt in LABEL_TYPES:
                lab_src = triplet[lt]
                rel_lab = os.path.relpath(lab_src, LABELS_TRAIN)
                lab_dst = os.path.join(LABELS_VAL, rel_lab)
                status_lab = move_file(lab_src, lab_dst, args.dry_run)
                if "moved" in status_lab:
                    labs_moved += 1

        overall["sequences"][seq] = {
            "bases_with_vis_nir": n_with_both,
            "eligible_with_3_labels": n_eligible,
            "planned_bases": target,
            "images_moved": imgs_moved,
            "labels_moved": labs_moved
        }
        overall["total_planned_bases"] += target
        overall["total_images_moved"] += imgs_moved
        overall["total_labels_moved"] += labs_moved

    print("=" * 78)
    print("Validation Split Creation Summary (paired VIS+NIR + 3 label formats)")
    print(f"ROOT: {ROOT}")
    print(f"FRACTION: {FRACTION:.2f} | SEED: {SEED} | DRY-RUN: {args.dry_run} | OVERWRITE: {OVERWRITE}")
    print("-" * 78)
    for seq, info in overall["sequences"].items():
        print(
            f"[{seq}] with_vis+nir={info.get('bases_with_vis_nir', 0)}, "
            f"eligible_3lbl={info.get('eligible_with_3_labels', 0)}, "
            f"planned_bases={info.get('planned_bases', 0)}, "
            f"imgs_moved={info.get('images_moved', 0)}, "
            f"labels_moved={info.get('labels_moved', 0)}"
            + (f" | note={info.get('note')}" if 'note' in info else "")
        )
    print("-" * 78)
    print(f"TOTAL bases with VIS+NIR: {overall['total_bases_with_vis_nir']}")
    print(f"TOTAL eligible (VIS+NIR+3 labels): {overall['total_eligible_with_3_labels']}")
    print(f"TOTAL planned bases for val: {overall['total_planned_bases']}")
    print(f"TOTAL images moved (VIS+NIR): {overall['total_images_moved']}")
    print(f"TOTAL labels moved (3 per base): {overall['total_labels_moved']}")
    print("=" * 78)
    if args.dry_run:
        print("\nNOTE: Dry run only. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
