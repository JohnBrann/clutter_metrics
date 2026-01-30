import argparse
import csv
import os
import re
import math
from dataclasses import dataclass
from glob import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


# ----------------------------
# Basic helpers
# ----------------------------

def imread_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def imwrite_rgb(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def safe_basename_no_ext(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def color_distance(c1, c2):
    a = np.array(c1, dtype=np.float32)
    b = np.array(c2, dtype=np.float32)
    return float(np.linalg.norm(a - b))


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Segment:
    color: tuple
    name: str
    mask: np.ndarray
    boundary_coords: np.ndarray   # Nx2 (y,x) int32


# ----------------------------
# Boundary sampling
# ----------------------------

def extract_boundary_coords(mask):
    """
    Returns boundary pixels only (Nx2 y,x).
    Avoids the interior-point fallback that can place points fully inside.
    """
    # Method A: erosion boundary
    # eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    # boundary = mask & (~eroded)
    # ys, xs = np.nonzero(boundary)

    # if len(ys) > 0:
    #     pts = np.stack([ys, xs], axis=1).astype(np.int32)
    #     return np.unique(pts, axis=0)

    # Method B: cheap "edge" test (4-neighborhood) if erosion fails
    # A pixel is boundary if it's in mask and has any neighbor not in mask.
    m = mask
    up    = np.zeros_like(m); up[1:, :]  = m[:-1, :]
    down  = np.zeros_like(m); down[:-1,:]= m[1:, :]
    left  = np.zeros_like(m); left[:,1:] = m[:, :-1]
    right = np.zeros_like(m); right[:,:-1]= m[:, 1:]

    interior = m & up & down & left & right
    boundary2 = m & (~interior)
    ys, xs = np.nonzero(boundary2)

    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    return np.unique(pts, axis=0)


def order_points_nearest_chain(pts):
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    ptsf = pts.astype(np.float64)
    if len(ptsf) == 1:
        return ptsf.copy()

    tree = cKDTree(ptsf[:, ::-1])  # (x,y)
    visited = np.zeros(len(ptsf), dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(1, len(ptsf)):
        last = order[-1]
        k = min(30, len(ptsf))
        _, idxs = tree.query(ptsf[last, ::-1], k=k)
        idxs = np.atleast_1d(idxs)

        nxt = None
        for idx in idxs:
            idx = int(idx)
            if not visited[idx]:
                nxt = idx
                break

        if nxt is None:
            unv = np.nonzero(~visited)[0]
            if unv.size == 0:
                break
            diffs = ptsf[unv] - ptsf[last]
            d2 = np.sum(diffs * diffs, axis=1)
            nxt = int(unv[int(np.argmin(d2))])

        order.append(nxt)
        visited[nxt] = True

    return ptsf[order]

def sample_points_along_chain(coords, spacing_px):
    if coords is None or len(coords) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pts = coords.astype(np.float64)

    if spacing_px <= 0:
        return np.unique(np.round(pts).astype(np.int32), axis=0)

    ordered = order_points_nearest_chain(pts)
    if len(ordered) < 2:
        return np.unique(np.round(ordered).astype(np.int32), axis=0)

    seg_dists = np.sqrt(np.sum(np.diff(ordered, axis=0) ** 2, axis=1))
    cumdist = np.concatenate(([0.0], np.cumsum(seg_dists)))
    total = cumdist[-1]

    if total <= 0:
        return np.unique(np.round(ordered).astype(np.int32), axis=0)

    num_samples = max(2, int(np.floor(total / spacing_px)) + 1)
    sample_ds = np.linspace(0.0, total, num=num_samples)

    xs = ordered[:, 1]
    ys = ordered[:, 0]
    samp_x = np.interp(sample_ds, cumdist, xs)
    samp_y = np.interp(sample_ds, cumdist, ys)

    samp = np.stack([samp_y, samp_x], axis=1)
    return np.unique(np.round(samp).astype(np.int32), axis=0)


# ----------------------------
# Segment extraction
# ----------------------------

def find_segments(img, min_pixels, spacing_px):
    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = tuple(map(int, colors[int(np.argmax(counts))]))

    segments = []
    for c in colors:
        color = tuple(map(int, c))
        if color == bg:
            continue

        mask = np.all(img == c, axis=2)
        if int(mask.sum()) < min_pixels:
            continue

        boundary = extract_boundary_coords(mask)
        sampled = sample_points_along_chain(boundary, spacing_px)
        if len(sampled) == 0:
            continue

        segments.append(Segment(color=color, name=rgb_to_hex(color), mask=mask, boundary_coords=sampled))

    segments.sort(key=lambda s: s.name)
    return segments


# ----------------------------
# Occlusion/color filtering
# ----------------------------

SCENE_VIEW_RE = re.compile(r"theta(\d+)_phi(\d+)", flags=re.IGNORECASE)

def parse_view_from_filename(filename):
    m = SCENE_VIEW_RE.search(os.path.basename(filename))
    if not m:
        return None
    return m.group(0)
    # return (int(m.group(1)), int(m.group(2)))

def load_colors_csv(colors_csv_path):
    mapping = {}
    if not os.path.isfile(colors_csv_path):
        return mapping

    with open(colors_csv_path, newline="") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            th = int(row.get("theta", 0))
            ph = int(row.get("phi", 0))
            fname = row.get("filename") or row.get("file") or ""
            rgb = (int(row.get("r", 0)), int(row.get("g", 0)), int(row.get("b", 0)))
            mapping[(th, ph, fname)] = rgb
            if (0, 0, fname) not in mapping:
                mapping[(0, 0, fname)] = rgb
    return mapping

def load_occlusion_csv(occl_csv_path, threshold):
    excluded = {}
    if not os.path.isfile(occl_csv_path):
        return excluded

    with open(occl_csv_path, newline="") as of:
        reader = csv.DictReader(of)
        for row in reader:
            if row.get("level", "").strip().lower() != "object":
                continue
            pct = float(row.get("occlusion_pct", 0.0))
            if pct < threshold:
                continue
            th = int(row.get("theta", 0))
            ph = int(row.get("phi", 0))
            fname = row.get("filename", "")
            excluded.setdefault((th, ph), set()).add(fname)
    return excluded

def build_excluded_colors_map(excluded_by_view, color_map):
    out = {}
    for (th, ph), fnames in excluded_by_view.items():
        s = set()
        for fname in fnames:
            if (th, ph, fname) in color_map:
                s.add(color_map[(th, ph, fname)])
            elif (0, 0, fname) in color_map:
                s.add(color_map[(0, 0, fname)])
        if s:
            out[(th, ph)] = s
    return out

def filter_segments_by_excluded_colors(segments, excluded_colors, color_tol):
    if not excluded_colors:
        return segments, []

    kept = []
    removed = []
    for seg in segments:
        drop = False
        for ex in excluded_colors:
            if color_tol == 0:
                if seg.color == ex:
                    drop = True
                    break
            else:
                if color_distance(seg.color, ex) <= float(color_tol):
                    drop = True
                    break
        if drop:
            removed.append(seg)
        else:
            kept.append(seg)
    return kept, removed


# ----------------------------
# Nearest-other distance computation
# ----------------------------

def compute_nearest_object_distances(segments):
    """
    For EACH boundary point of EACH segment:
      connect to the nearest boundary point on ANY OTHER segment.

    This guarantees every point gets a connection if there is at least
    one other segment with boundary points.
    """
    if len(segments) < 2:
        return []

    # Flatten all points + owners
    all_pts_list = []
    owners = []
    for i, seg in enumerate(segments):
        pts = seg.boundary_coords
        if pts is None or len(pts) == 0:
            continue
        all_pts_list.append(pts)
        owners.extend([i] * len(pts))

    if not all_pts_list:
        return []

    all_pts = np.vstack(all_pts_list).astype(np.int32)
    owners = np.array(owners, dtype=np.int32)

    rows = []

    # For each segment, build a KD-tree of "all other segments" once
    for i, seg in enumerate(segments):
        src_pts = seg.boundary_coords
        if src_pts is None or len(src_pts) == 0:
            continue

        other_mask = owners != i
        other_pts = all_pts[other_mask]

        # If there are no other points, no connections possible
        if other_pts.shape[0] == 0:
            continue

        other_tree = cKDTree(other_pts[:, ::-1].astype(np.float64))  # (x,y)

        for (sy, sx) in src_pts:
            dist, idx = other_tree.query([float(sx), float(sy)], k=1)
            ty, tx = other_pts[int(idx)]
            rows.append((i, -1, float(dist), int(sy), int(sx), int(ty), int(tx)))

    # NOTE: tgt_idx is set to -1 above because we didn't track target segment id.
    # If you want target_obj/target_idx in the CSV, we can add it back cleanly.
    return rows



# ----------------------------
# Visualization (NEW)
# ----------------------------

def visualize_connections(img, segments, rows, out_path):
    """
    Saves a PNG showing:
      - image
      - sampled boundary points
      - all nearest-neighbor connection lines (GREEN)
    """
    H, W, _ = img.shape
    fig, ax = plt.subplots(figsize=(max(6, W / 60), max(6, H / 60)), dpi=150)
    ax.imshow(img)

    for seg in segments:
        pts = seg.boundary_coords
        if pts is None or len(pts) == 0:
            continue
        ax.scatter(pts[:, 1], pts[:, 0], s=6, alpha=0.9, linewidths=0)

    # GREEN connection lines
    for (src_i, tgt_i, dist, sy, sx, ty, tx) in rows:
        ax.plot([sx, tx], [sy, ty], linewidth=0.6, alpha=0.8, color="green")

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


# ----------------------------
# Per-image processing
# ----------------------------

def process_single_image(img_path, out_dir, spacing_px, min_pixels, excluded_colors_for_view, color_tol):
    img = imread_rgb(img_path)
    segments_all = find_segments(img, min_pixels=min_pixels, spacing_px=spacing_px)

    segments, removed = filter_segments_by_excluded_colors(
        segments_all, excluded_colors_for_view, color_tol
    )

    # Mask excluded segments to black
    masked = img.copy()
    for seg in removed:
        masked[seg.mask] = np.array([0, 0, 0], dtype=np.uint8)

    base = safe_basename_no_ext(img_path)

    masked_out = os.path.join(out_dir, f"{base}_masked.png")
    imwrite_rgb(masked_out, masked)

    # Compute distances and write CSV + trailing average row
    rows = compute_nearest_object_distances(segments)
    csv_out = os.path.join(out_dir, f"{base}_distances.csv")

    dists = [r[2] for r in rows]
    avg = float(np.mean(dists)) if len(dists) > 0 else float("nan")

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        # w.writerow(["source_obj", "source_idx", "target_obj", "target_idx",
        #             "distance_px", "src_x", "src_y", "tgt_x", "tgt_y"])
        w.writerow(["source_obj", "source_idx","distance_px"])

        for (src_i, tgt_i, dist, sy, sx, ty, tx) in rows:
            # w.writerow([
            #     segments[src_i].name, src_i,
            #     segments[tgt_i].name, tgt_i,
            #     f"{dist:.3f}",
            #     sx, sy, tx, ty
            # ])
            w.writerow([
                segments[src_i].name, src_i,
                f"{dist:.3f}",
            ])

        w.writerow([])
        w.writerow(["AVG_DISTANCE_PX", f"{avg:.3f}"])

    #  save visualization
    viz_out = os.path.join(out_dir, f"{base}_distances_viz.png")
    visualize_connections(masked, segments, rows, viz_out)

    return avg


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch compute 2D perimeter-corrected distances for segmented images in a folder.")
    parser.add_argument("--dataset-name", required=True, help="Dataset folder name under ./data/<dataset_name>/")
    parser.add_argument("--spacing", type=float, default=16.0, help="Distance between sampled boundary points in pixels (default: 16.0).")
    parser.add_argument("--min-pixels", type=int, default=5, help="Minimum pixels for an object segment to be considered (default: 5).")
    parser.add_argument("--skip-patterns", default="_distances,_metrics,_masked", help="Comma-separated substrings; files containing any will be skipped (default: '_distances,_metrics,_masked').")
    parser.add_argument("--occlusion-threshold", type=float, default=50.0, help="Occlusion threshold pct: objects with occlusion_pct >= this are excluded (default 50.0).")
    parser.add_argument("--occlusion-color-tol", type=int, default=0, help="Color distance tolerance (Euclidean) used when comparing segment color to excluded colors (default 0 = exact match).")
    args = parser.parse_args()

    base_dir = os.path.join("..", "data", "replica", args.dataset_name)
    input_dir = os.path.join(base_dir, "scene_groundtruths")
    out_dir = os.path.join(base_dir, "distance")
    os.makedirs(out_dir, exist_ok=True)

    occl_csv = os.path.join(base_dir, "occlusion", "per_object_occlusion.csv")
    colors_csv = os.path.join(base_dir, "occlusion", "per_object_colors.csv")

    color_map = load_colors_csv(colors_csv)
    excluded_by_view = load_occlusion_csv(occl_csv, args.occlusion_threshold)
    excluded_colors_map = build_excluded_colors_map(excluded_by_view, color_map)

    skip_subs = [s.strip() for s in args.skip_patterns.split(",") if s.strip()]

    files = sorted(glob(os.path.join(input_dir, "*.png")))
    processed = 0

    view_avgs = []   # list of (view, avg_distance)

    for fpath in files:
        base = os.path.basename(fpath)
        if any(sub in base for sub in skip_subs):
            continue

        view = parse_view_from_filename(base)
        excluded_colors_for_view = excluded_colors_map.get(view, None) if view else None

        avg_distance = process_single_image(
            fpath,
            out_dir=out_dir,
            spacing_px=args.spacing,
            min_pixels=args.min_pixels,
            excluded_colors_for_view=excluded_colors_for_view,
            color_tol=args.occlusion_color_tol,
        )

        processed += 1

        # if not math.isnan(avg_distance):
        view_avgs.append((view, avg_distance))

    # ---- write single summary CSV ----
    summary_csv = os.path.join(out_dir, "distance_summary.csv")

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["viewpoint", "avg_distance"])

        for view, avg in view_avgs:
            w.writerow([view, f"{avg:.3f}"])

        # scene average (mean of per-view avgs)
        scene_avg = sum(a for _, a in view_avgs) / len(view_avgs)

        w.writerow(["full_scene", f"{scene_avg:.3f}"])

    print(f"Processed {processed} images.")
    print(f"Summary CSV written to: {summary_csv}")


if __name__ == "__main__":
    main()
