import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mplcursors


REPLICA_ROOT = Path("../data/replica")
REL_CSV_PATH = Path("distance") / "theta*_phi*_*.csv"
OUTPUT_CSV = Path("proximity_mnet_all_scene_avg_total_connections.csv")


def _count_connections_in_file(csv_file: Path) -> int:
    lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()

    # drop trailing empty/whitespace-only lines
    while lines and not lines[-1].strip():
        lines.pop()

    # if file is too small to contain header + summary, treat as 0
    if len(lines) < 2:
        return 0

    # exclude last line
    lines_wo_last = lines[:-1]

    # exclude header (first line)
    data_lines = lines_wo_last[1:]

    # "connections" = number of remaining lines
    return max(0, len(data_lines))


def collect_connections_data(root: Path = REPLICA_ROOT, rel_csv: Path = REL_CSV_PATH):
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

        # find all viewpoint CSVs for this scene
        csv_files = sorted(scene_dir.glob(str(rel_csv)))
        if not csv_files:
            continue

        counts = []
        for f in csv_files:
            c = _count_connections_in_file(f)
            if c > 0:
                counts.append(c)

        if not counts:
            print(f"Skipping scene {scene_number} (no valid connection counts)")
            continue

        counts.sort()
        n = len(counts)

        min_c = counts[0]
        max_c = counts[-1]
        avg_c = sum(counts) / n

        if n % 2 == 1:
            median_c = counts[n // 2]
        else:
            median_c = (counts[n // 2 - 1] + counts[n // 2]) / 2.0

        var = sum((c - avg_c) ** 2 for c in counts) / n
        std_c = math.sqrt(var)

        results.append((
            scene_number,
            round(min_c, 0),
            round(median_c, 0),
            round(max_c, 0),
            round(std_c, 2),
            round(avg_c, 2),
        ))

    results.sort(key=lambda x: x[0])
    return results


def plot_information(scene_connections):
    if not scene_connections:
        print("No connection data found to plot.")
        return

    easy_scenes, easy_vals = [], []
    med_scenes, med_vals = [], []
    hard_scenes, hard_vals = [], []

    for scene, min_c, median_c, max_c, std_c, avg_c in scene_connections:
        if scene < 100:
            easy_scenes.append(scene)
            easy_vals.append(avg_c)
        elif scene < 200:
            med_scenes.append(scene)
            med_vals.append(avg_c)
        else:
            hard_scenes.append(scene)
            hard_vals.append(avg_c)


    print(f'easy avg: {sum(easy_vals)/len(easy_vals)}')
    print(f'medium avg: {sum(med_vals)/len(med_vals)}')
    print(f'hard avg: {sum(hard_vals)/len(hard_vals)}')
    
    fig, ax = plt.subplots()

    ax.axvspan(0, 99, alpha=0.08)
    ax.axvspan(100, 199, alpha=0.08)
    ax.axvspan(200, 299, alpha=0.08)

    sc_easy = ax.scatter(easy_scenes, easy_vals, label="easy (0–99)", s=30)
    sc_med  = ax.scatter(med_scenes,  med_vals,  label="medium (100–199)", s=30)
    sc_hard = ax.scatter(hard_scenes, hard_vals, label="hard (200–299)", s=30)

    ax.set_title("SceneReplica Avg. Total Connections Per Scene")
    ax.set_xlabel("Scene number")
    ax.set_ylabel("Avg total connections (per viewpoint CSV)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

    cursor = mplcursors.cursor([sc_easy, sc_med, sc_hard], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"Scene {int(x)}\navg total connections: {y:.2f}")

    plt.tight_layout()
    plt.show()


def save_to_csv(scene_connections, output_path: Path = OUTPUT_CSV):
    if not scene_connections:
        print("No data to save.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_number",
            "min_connections",
            "median_connections",
            "max_connections",
            "std_dev",
            "avg_connections"
        ])
        for row in scene_connections:
            writer.writerow(list(row))

    print(f"Saved {len(scene_connections)} rows to {output_path}")


def main():
    data = collect_connections_data()
    save_to_csv(data)
    plot_information(data)


if __name__ == "__main__":
    main()