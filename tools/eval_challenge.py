import os
import json
import argparse
import glob
import traceback
import math
import numpy as np
import trimesh
from scipy.spatial import distance as compute_dist_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

# ==============================================================================
# Metric Calculation Functions (Adapted from User's Script)
# ==============================================================================


def compute_tooth_size(points, centroid):
    size = np.sqrt(np.sum((centroid - points) ** 2, axis=0))
    return size


def calculate_jaw_TSA(gt_instances, pred_instances):
    """
    Teeth segmentation accuracy (TSA): Binary F1-score (tooth vs background).
    """
    gt_binary = (gt_instances != 0).astype(int)
    pred_binary = (pred_instances != 0).astype(int)
    return f1_score(gt_binary, pred_binary, average="micro")


def extract_centroids(instance_label_dict):
    centroids_list = []
    for k, v in instance_label_dict.items():
        centroids_list.append((v["centroid"]))
    return centroids_list


def centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict):
    gt_cent_list = extract_centroids(gt_instance_label_dict)
    pred_cent_list = extract_centroids(pred_instance_label_dict)

    if not gt_cent_list or not pred_cent_list:
        return {}

    M = compute_dist_matrix.cdist(gt_cent_list, pred_cent_list)
    row_ind, col_ind = linear_sum_assignment(M)

    gt_keys = list(gt_instance_label_dict.keys())
    pred_keys = list(pred_instance_label_dict.keys())

    matching_dict = {gt_keys[i]: pred_keys[j] for i, j in zip(row_ind, col_ind)}
    return matching_dict


def calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict):
    TLA = 0
    if len(gt_instance_label_dict) == 0:
        return 0.0

    for inst, info in gt_instance_label_dict.items():
        if inst in matching_dict.keys():
            TLA += np.linalg.norm(
                (
                    gt_instance_label_dict[inst]["centroid"]
                    - pred_instance_label_dict[matching_dict[inst]]["centroid"]
                )
                / gt_instance_label_dict[inst]["tooth_size"]
            )
        else:
            TLA += 5 * np.linalg.norm(gt_instance_label_dict[inst]["tooth_size"])

    return TLA / len(gt_instance_label_dict.keys())


def calculate_jaw_TIR(
    gt_instance_label_dict, pred_instance_label_dict, matching_dict, threshold=0.5
):
    if len(matching_dict) == 0:
        return 0.0
    tir = 0
    for gt_inst, pred_inst in matching_dict.items():
        dist = np.linalg.norm(
            (
                gt_instance_label_dict[gt_inst]["centroid"]
                - pred_instance_label_dict[pred_inst]["centroid"]
            )
            / gt_instance_label_dict[gt_inst]["tooth_size"]
        )

        if (
            dist < threshold
            and gt_instance_label_dict[gt_inst]["label"]
            == pred_instance_label_dict[pred_inst]["label"]
        ):
            tir += 1
    return tir / len(matching_dict)


def calculate_metrics_for_sample(gt_label_dict, pred_label_dict):
    gt_instances = np.array(gt_label_dict["instances"])
    gt_labels = np.array(gt_label_dict["labels"])

    # GT Processing
    u_instances = np.unique(gt_instances)
    u_instances = u_instances[u_instances != 0]

    gt_instance_label_dict = {}
    for l in u_instances:
        mask = gt_instances == l
        gt_lbl = gt_labels[mask]
        label = np.unique(gt_lbl)
        if len(label) == 1:
            gt_verts = gt_label_dict["mesh_vertices"][mask]
            gt_center = np.mean(gt_verts, axis=0)
            tooth_size = compute_tooth_size(gt_verts, gt_center)
            gt_instance_label_dict[str(l)] = {
                "label": label[0],
                "centroid": gt_center,
                "tooth_size": tooth_size,
            }

    # Pred Processing
    pred_labels = np.array(pred_label_dict["labels"])
    pred_instances = pred_labels.copy()

    # Build pred dict
    pred_instance_label_dict = {}
    u_pred_instances = np.unique(pred_instances)
    u_pred_instances = u_pred_instances[u_pred_instances != 0]

    for pred_inst in u_pred_instances:
        label_id = pred_inst
        mask = pred_instances == pred_inst

        pred_verts = gt_label_dict["mesh_vertices"][mask]
        if len(pred_verts) == 0:
            continue

        pred_center = np.mean(pred_verts, axis=0)
        pred_instance_label_dict[str(pred_inst)] = {
            "label": label_id,
            "centroid": pred_center,
        }

    # Calculate
    matching_dict = centroids_pred_to_gt_attribution(
        gt_instance_label_dict, pred_instance_label_dict
    )

    try:
        jaw_TLA = calculate_jaw_TLA(
            gt_instance_label_dict, pred_instance_label_dict, matching_dict
        )
    except Exception as e:
        jaw_TLA = 0

    try:
        jaw_TSA = calculate_jaw_TSA(gt_instances, pred_instances)
    except Exception as e:
        jaw_TSA = 0

    try:
        jaw_TIR = calculate_jaw_TIR(
            gt_instance_label_dict, pred_instance_label_dict, matching_dict
        )
    except Exception as e:
        jaw_TIR = 0

    return jaw_TLA, jaw_TSA, jaw_TIR


# ==============================================================================
# Visualization Helpers
# ==============================================================================


def get_color_map(num_classes=33, seed=42):
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(num_classes + 1, 3))
    colors[0] = [200, 200, 200]  # Background (Gingiva) = Grey/White/Pink
    # Make sure label 11-18, 21-28 etc distinct
    return colors


def export_mesh(mesh, labels, save_path, color_map):
    # Ensure labels are within range
    labels = labels.astype(int)

    # Map labels to colors
    # labels can check range
    vertex_colors = np.zeros((len(labels), 3), dtype=np.uint8)

    # If labels match color map range
    valid_mask = labels < len(color_map)
    vertex_colors[valid_mask] = color_map[labels[valid_mask]]

    # Create new mesh for export
    out_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    out_mesh.export(save_path)


# ==============================================================================
# Main Processing Logic
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp_root",
        default="./exp/teeth3ds",
        help="Experiment root containing fold_X directories",
    )
    p.add_argument("--obj_dir", default="/path/to/data/data_obj_parent_directory")
    p.add_argument("--json_dir", default="/path/to/data/data_json_parent_directory")
    p.add_argument(
        "--save_mesh", action="store_true", default=True, help="Save colored PLY meshes"
    )
    p.add_argument(
        "--save_mesh_dir", default="vis_meshes", help="Directory name for saved meshes"
    )
    return p.parse_args()


def main():
    args = parse_args()

    folds = [0, 1, 2, 3, 4]
    color_map = get_color_map(50)  # Enough for label 48

    global_TLA = []
    global_TSA = []
    global_TIR = []

    print(f"{'Sample':<30} | {'TSA':<8} | {'TLA':<8} | {'TIR':<8} |")
    print("-" * 65)

    for fold in folds:
        result_dir = os.path.join(args.exp_root, f"fold{fold}", "result")
        pred_files = sorted(glob.glob(os.path.join(result_dir, "*_pred.npy")))

        if not pred_files:
            continue

        print(f"Processing Fold {fold} ({len(pred_files)} samples)...")

        # Create output dir for meshes
        if args.save_mesh:
            vis_dir = os.path.join(args.exp_root, args.save_mesh_dir)
            os.makedirs(vis_dir, exist_ok=True)

        for pred_path in pred_files:
            basename = os.path.basename(pred_path)
            name_part = basename.replace("_pred.npy", "")
            patient, jaw = name_part.rsplit("_", 1)

            # Load Prediction
            pred_labels = np.load(pred_path)

            # Load Original OBJ and JSON
            obj_path = os.path.join(args.obj_dir, patient, f"{patient}_{jaw}.obj")
            json_path = os.path.join(args.json_dir, patient, f"{patient}_{jaw}.json")

            if not os.path.exists(obj_path) or not os.path.exists(json_path):
                print(f"Missing data for {name_part}, skipping.")
                continue

            mesh = trimesh.load(obj_path, process=False, force="mesh")
            vertices = np.array(mesh.vertices)

            with open(json_path, "r") as f:
                gt_data = json.load(f)

            if len(pred_labels) != len(vertices):
                print(
                    f"Size mismatch for {name_part}: Pred {len(pred_labels)} != Mesh {len(vertices)}. Skipping."
                )
                continue

            # Metric Calculation
            # Remap predictions to FDI labels (0-32 -> 11-48)
            # 0: gingiva, 11-18: Q1, 21-28: Q2, 31-38: Q3, 41-48: Q4
            _FDI_LABELS = (
                [0]
                + list(range(11, 19))
                + list(range(21, 29))
                + list(range(31, 39))
                + list(range(41, 49))
            )
            pred_fdi = np.array(_FDI_LABELS)[pred_labels]

            gt_label_dict = {
                "instances": gt_data["instances"],
                "labels": gt_data["labels"],
                "mesh_vertices": vertices,
            }
            # Use FDI labels for metric calculation
            pred_label_dict = {"instances": pred_fdi, "labels": pred_fdi}

            tla, tsa, tir = calculate_metrics_for_sample(gt_label_dict, pred_label_dict)

            global_TLA.append(tla)
            global_TSA.append(tsa)
            global_TIR.append(tir)

            # Save Mesh
            if args.save_mesh:
                # Save as PLY: patient_jaw_pred.ply
                save_name = f"{name_part}_pred.ply"
                save_path = os.path.join(vis_dir, save_name)
                export_mesh(mesh, pred_labels, save_path, color_map)

    print("-" * 65)
    print("Final Average Results (All Folds):")
    if global_TSA:
        print(f"TSA (F1) : {np.mean(global_TSA):.4f} +/- {np.std(global_TSA):.4f}")
        print(f"TLA      : {np.mean(global_TLA):.4f} +/- {np.std(global_TLA):.4f}")
        print(f"TIR      : {np.mean(global_TIR):.4f} +/- {np.std(global_TIR):.4f}")
        score = (
            np.mean(global_TSA) + math.exp(-np.mean(global_TLA)) + np.mean(global_TIR)
        ) / 3
        print(f"Global Score: {score:.4f}")
    else:
        print("No samples processed.")


if __name__ == "__main__":
    main()
