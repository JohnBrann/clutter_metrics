import os
from pathlib import Path
import mplcursors
import matplotlib.pyplot as plt
import csv
import math 


REPLICA_ROOT = Path("../data/replica")
REL_CSV_PATH = Path("occlusion") / "occlusion_summary.csv"
OUTPUT_CSV = Path("occlusion_mnet_all_scene_avg.csv")


def collect_occlusion_data(root: Path = REPLICA_ROOT, rel_csv: Path = REL_CSV_PATH):
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

        # Collect all numeric occlusion values (skip header)
        occlusions = []
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("viewpoint,") or line.lower().startswith("full_scene,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                occlusions.append(float(parts[-1]))
            except ValueError:
                continue

        if not occlusions:
            print(f"Skipping {csv_path} (no valid data lines)")
            continue

        # ---- Summary statistics ----
        occlusions.sort()

        min_o = occlusions[0]
        max_o = occlusions[-1]
        avg_o = sum(occlusions) / len(occlusions)

        # Median (NEW)
        n = len(occlusions)
        if n % 2 == 1:
            median_o = occlusions[n // 2]
        else:
            median_o = (occlusions[n // 2 - 1] + occlusions[n // 2]) / 2.0

        # Population standard deviation
        var = sum((o - avg_o) ** 2 for o in occlusions) / n
        std_o = math.sqrt(var)

        results.append((
            scene_number,
            round(min_o, 2),
            round(median_o, 2),
            round(max_o, 2),
            round(std_o, 2),
            round(avg_o, 2),
        ))

    results.sort(key=lambda x: x[0])
    return results


def plot_information(scene_occlusions):
    if not scene_occlusions:
        print("No occlusion data found to plot.")
        return

    easy_scenes, easy_occ = [], []
    med_scenes, med_occ = [], []
    hard_scenes, hard_occ = [], []

    # Use average for plotting 
    for scene, min_o, median_o, max_o, std_o, avg_o in scene_occlusions:
        occ = avg_o
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

    ax.axvspan(0, 99, alpha=0.08)
    ax.axvspan(100, 199, alpha=0.08)
    ax.axvspan(200, 299, alpha=0.08)

    sc_easy = ax.scatter(easy_scenes, easy_occ, label="easy (0–99)", s=30)
    sc_med  = ax.scatter(med_scenes, med_occ, label="medium (100–199)", s=30)
    sc_hard = ax.scatter(hard_scenes, hard_occ, label="hard (200–299)", s=30)

    ax.set_title("Occlusion Scores Per ManipulationNet Scene")
    ax.set_xlabel("Scene number")
    ax.set_ylabel("Avg Occlusion (%)")

    ax.set_ylim(0, 50)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

    cursor = mplcursors.cursor([sc_easy, sc_med, sc_hard], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"Scene {int(x)}\nOcclusion: {y:.2f}%")

    plt.tight_layout()
    plt.show()


def save_to_csv(scene_occlusion, output_path: Path = OUTPUT_CSV):
    if not scene_occlusion:
        print("No data to save.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_number",
            "min_occlusion",
            "median_occlusion",
            "max_occlusion",
            "std_dev",
            "avg_occlusion"
        ])
        for scene, min_o, median_o, max_o, std_o, avg_o in scene_occlusion:
            writer.writerow([scene, min_o, median_o, max_o, std_o, avg_o])

    print(f"Saved {len(scene_occlusion)} rows to {output_path}")


def main():
    data = collect_occlusion_data()
    save_to_csv(data)
    plot_information(data)


if __name__ == "__main__":
    main()