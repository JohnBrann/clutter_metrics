import os
from pathlib import Path
import mplcursors
import matplotlib.pyplot as plt


REPLICA_ROOT = Path("../data/replica")
REL_CSV_PATH = Path("occlusion") / "occlusion_summary.csv"


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
            # Uncomment if you want to see what's missing:
            # print(f"Missing: {csv_path}")
            continue

        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()

        last = None
        for line in reversed(lines):
            line = line.strip()
            if line and not line.lower().startswith("viewpoint,"):
                last = line
                break

        if not last:
            print(f"Skipping {csv_path} (no data lines)")
            continue

        parts = [p.strip() for p in last.split(",")]
        try:
            occlusion = float(parts[-1])
        except ValueError:
            print(f"Skipping {csv_path} (bad last line: {last!r})")
            continue

        results.append((scene_number, occlusion))

    results.sort(key=lambda x: x[0])
    return results



def plot_information(scene_occlusions):

    if not scene_occlusions:
        print("No occlusion data found to plot.")
        return

    easy_scenes, easy_occ = [], []
    med_scenes, med_occ = [], []
    hard_scenes, hard_occ = [], []

    for scene, occ in scene_occlusions:
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

    ax.set_title("SceneReplica Occlusion Score")
    ax.set_xlabel("Scene number")
    ax.set_ylabel("Occlusion (%)")

    ax.set_ylim(0, 100)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

    # ---- Hover tooltips ----
    cursor = mplcursors.cursor([sc_easy, sc_med, sc_hard], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"Scene {int(x)}\nOcclusion: {y:.2f}%")

    plt.tight_layout()
    plt.show()

def main():
    data = collect_occlusion_data()
    plot_information(data)


if __name__ == "__main__":
    main()
