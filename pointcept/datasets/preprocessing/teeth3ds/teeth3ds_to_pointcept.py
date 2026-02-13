"""
Convert Teeth3DS dataset to Pointcept directory format with 5-fold CV splits.

Supports two source modes:
  1. --from_npz <dir>   : Read from existing .npz files (faster, no trimesh needed)
  2. Default            : Read from OBJ + JSON directly

Output:
    <output_dir>/fold_0/<patient>_<jaw>/{coord,color,normal,segment}.npy
    ...

# FDI label -> contiguous id (33 classes)
# 0: gingiva, 11-18: Q1 upper right, 21-28: Q2 upper left
# 31-38: Q3 lower left, 41-48: Q4 lower right

Usage:
    # From existing npz (fast)
    python teeth3ds_to_pointcept.py --from_npz /path/to/sonata_format

    # From OBJ+JSON (original)
    python teeth3ds_to_pointcept.py --obj_dir ... --json_dir ...
"""

import os
import json
import argparse
import random

import numpy as np

# FDI label -> contiguous id (33 classes)
# 0: gingiva, 11-18: Q1 upper right, 21-28: Q2 upper left
# 31-38: Q3 lower left, 41-48: Q4 lower right
_FDI_LABELS = (
    [0]
    + list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)
LABEL_REMAP = {fdi: idx for idx, fdi in enumerate(_FDI_LABELS)}
NUM_CLASSES = len(_FDI_LABELS)  # 33

CLASS_NAMES = ["gingiva"] + [f"tooth_{l}" for l in _FDI_LABELS[1:]]


def parse_args():
    p = argparse.ArgumentParser(description="Teeth3DS -> Pointcept converter")
    p.add_argument(
        "--from_npz",
        type=str,
        default=None,
        help="Path to existing sonata-format npz directory (skip OBJ/JSON processing)",
    )
    p.add_argument(
        "--obj_dir",
        default="/path/to/data/data_obj_parent_directory",
    )
    p.add_argument(
        "--json_dir",
        default="/path/to/data/data_json_parent_directory",
    )
    p.add_argument(
        "--output_dir",
        default="./data/teeth3ds",
    )
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def remap_labels(labels_raw):
    """Remap FDI labels to contiguous 0-based ids."""
    out = np.full_like(labels_raw, fill_value=-1, dtype=np.int32)
    for src, dst in LABEL_REMAP.items():
        out[labels_raw == src] = dst
    n_unknown = (out == -1).sum()
    if n_unknown > 0:
        print(
            f"  Warning: {n_unknown} vertices with unknown labels, set to -1 (ignore)"
        )
    return out


def save_sample(coord, color, normal, segment_raw, out_dir):
    """Remap labels and save as individual npy files."""
    segment = remap_labels(segment_raw)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coord.npy"), coord.astype(np.float32))
    np.save(os.path.join(out_dir, "color.npy"), color.astype(np.float32))
    np.save(os.path.join(out_dir, "normal.npy"), normal.astype(np.float32))
    np.save(os.path.join(out_dir, "segment.npy"), segment)


def process_from_npz(npz_dir, output_dir, n_folds, seed):
    """Convert from existing Sonata-format npz files."""
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    print(f"Found {len(npz_files)} npz files")

    # Extract patient names (each patient has _upper.npz and _lower.npz)
    patients = sorted(set(f.rsplit("_", 1)[0] for f in npz_files))
    print(f"Found {len(patients)} patients")

    # Assign patients to folds
    random.seed(seed)
    shuffled = patients.copy()
    random.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(shuffled):
        folds[i % n_folds].append(p)

    patient_to_fold = {}
    for fi, fold_patients in enumerate(folds):
        print(f"Fold {fi}: {len(fold_patients)} patients")
        for p in fold_patients:
            patient_to_fold[p] = fi

    # Convert
    success = 0
    for i, npz_name in enumerate(npz_files):
        sample_name = npz_name[:-4]  # remove .npz
        patient = sample_name.rsplit("_", 1)[0]
        fold_idx = patient_to_fold[patient]
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        out_dir = os.path.join(fold_dir, sample_name)

        data = dict(np.load(os.path.join(npz_dir, npz_name)))
        save_sample(
            data["coord"], data["color"], data["normal"], data["segment"], out_dir
        )
        success += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(npz_files)} done")

    print(f"\nDone! {success} samples saved to {output_dir}")


def process_from_obj_json(obj_dir, json_dir, output_dir, n_folds, seed):
    """Convert from original OBJ + JSON files."""
    import trimesh

    patients = sorted(os.listdir(json_dir))
    print(f"Found {len(patients)} patients")

    random.seed(seed)
    shuffled = patients.copy()
    random.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(shuffled):
        folds[i % n_folds].append(p)

    for fi, fold in enumerate(folds):
        print(f"Fold {fi}: {len(fold)} patients")

    success = 0
    skipped = 0
    for fi, fold_patients in enumerate(folds):
        fold_dir = os.path.join(output_dir, f"fold_{fi}")
        for pi, patient in enumerate(fold_patients):
            for jaw in ("upper", "lower"):
                obj_path = os.path.join(obj_dir, patient, f"{patient}_{jaw}.obj")
                json_path = os.path.join(json_dir, patient, f"{patient}_{jaw}.json")
                if not os.path.exists(obj_path) or not os.path.exists(json_path):
                    skipped += 1
                    continue

                mesh = trimesh.load(obj_path, process=False)
                coord = np.array(mesh.vertices, dtype=np.float32)
                color = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
                normal = np.array(mesh.vertex_normals, dtype=np.float32)

                with open(json_path, "r") as f:
                    ann = json.load(f)
                labels_raw = np.array(ann["labels"], dtype=np.int64)
                assert coord.shape[0] == labels_raw.shape[0]

                out_dir = os.path.join(fold_dir, f"{patient}_{jaw}")
                save_sample(coord, color, normal, labels_raw, out_dir)
                success += 1

            if (pi + 1) % 20 == 0:
                print(f"  Fold {fi}: {pi+1}/{len(fold_patients)} patients done")

    print(f"\nDone! {success} samples saved, {skipped} skipped.")
    print(f"Output: {output_dir}")


def main():
    args = parse_args()
    if args.from_npz:
        process_from_npz(args.from_npz, args.output_dir, args.n_folds, args.seed)
    else:
        process_from_obj_json(
            args.obj_dir, args.json_dir, args.output_dir, args.n_folds, args.seed
        )


if __name__ == "__main__":
    main()
