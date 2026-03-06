import os
from pathlib import Path
import mplcursors
import matplotlib.pyplot as plt
import csv
import math 


REPLICA_ROOT = Path("../data/replica")
REL_CSV_PATH = Path("distance") / "distance_summary.csv"
OUTPUT_CSV = Path("proximity_mnet_all_scene_avg.csv")


def collect_distance_data(root: Path = REPLICA_ROOT, rel_csv: Path = REL_CSV_PATH):
    results = []

    if not root.exists():
        raise FileNotFoundError(f"Replica root not found: {root}")

    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue

        try:
            scene_number = int(scene_dir.name)
        except ValueError:
            continue

        csv_path = scene_dir / rel_csv
        if not csv_path.exists():
            continue

        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()

        #  collect all numeric distances in the scene file (skip header)
        distances = []
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("viewpoint,") or line.lower().startswith("full_scene,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                distances.append(float(parts[-1]))
            except ValueError:
                continue

        if not distances:
            print(f"Skipping {csv_path} (no valid data lines)")
            continue

        # compute summary stats for the scene
        distances.sort()

        min_d = distances[0]
        max_d = distances[-1]
        avg_d = sum(distances) / len(distances)

        # Medians
        n = len(distances)
        if n % 2 == 1:
            median_d = distances[n // 2]
        else:
            median_d = (distances[n // 2 - 1] + distances[n // 2]) / 2.0

        var = sum((d - avg_d) ** 2 for d in distances) / n  # population variance
        std_d = math.sqrt(var)

        results.append((
            scene_number,
            round(min_d, 2),
            round(median_d, 2),
            round(max_d, 2),
            round(std_d, 2),
            round(avg_d, 2),
        ))

    results.sort(key=lambda x: x[0])
    return results


def plot_information(scene_distances):
    if not scene_distances:
        print("No distance data found to plot.")
        return

    easy_scenes, easy_occ = [], []
    med_scenes, med_occ = [], []
    hard_scenes, hard_occ = [], []

    #  unpack avg distance from the tuple
    for scene, min_d, median_d, max_d, std_d, avg_d in scene_distances:
        occ = avg_d
        if scene < 100:
            easy_scenes.append(scene)
            easy_occ.append(occ)
        elif scene < 200:
            med_scenes.append(scene)
            med_occ.append(occ)
        else:
            hard_scenes.append(scene)
            hard_occ.append(occ)

    fig, ax = plt.subplots()

    # Difficulty bands
    ax.axvspan(0, 99, alpha=0.08)
    ax.axvspan(100, 199, alpha=0.08)
    ax.axvspan(200, 299, alpha=0.08)

    # Scatter plots
    sc_easy = ax.scatter(easy_scenes, easy_occ, label="easy (0–99)", s=30)
    sc_med  = ax.scatter(med_scenes, med_occ, label="medium (100–199)", s=30)
    sc_hard = ax.scatter(hard_scenes, hard_occ, label="hard (200–299)", s=30)

    ax.set_title("SceneReplica Avg. Distance Scores Per Scene")
    ax.set_xlabel("Scene number")
    ax.set_ylabel("Avg distance (pixel units)")

    ax.set_ylim(100, 0)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

    # ---- Hover tooltips ----
    cursor = mplcursors.cursor([sc_easy, sc_med, sc_hard], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"Scene {int(x)}\ndistance: {y:.2f}")

    plt.tight_layout()
    plt.show()


def save_to_csv(scene_distances, output_path: Path = OUTPUT_CSV):
    if not scene_distances:
        print("No data to save.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_number",
            "min_proximity",
            "median_proximity",
            "max_proximity",
            "std_dev",
            "avg_proximity"
        ])
        for scene, min_d, median_d, max_d, std_d, avg_d in scene_distances:
            writer.writerow([scene, min_d, median_d, max_d, std_d, avg_d])

    print(f"Saved {len(scene_distances)} rows to {output_path}")


def main():
    data = collect_distance_data()
    save_to_csv(data)
    plot_information(data)


if __name__ == "__main__":
    main()