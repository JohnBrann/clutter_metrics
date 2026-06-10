#!/usr/bin/env python3
import os
import csv
import json
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

SCENES_DIR = "/home/csrobot/clutter_quantification/scenes"
PORT = 8000

# Matches {any_prefix}_{zero-padded-number}, e.g. batch1_001
SCENE_PATTERN = re.compile(r'^(.+)_(\d+)$')


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/scenes":
            self.serve_json(get_scenes())

        elif parsed.path.startswith("/api/scene/"):
            scene_id = parsed.path.split("/api/scene/")[-1]
            self.serve_json(get_scene_detail(scene_id))

        elif parsed.path.startswith("/api/image/"):
            parts = parsed.path.split("/api/image/")[-1].split("/")
            if len(parts) == 2:
                scene_id, filename = parts
                img_path = os.path.join(SCENES_DIR, scene_id, filename)
                if os.path.isfile(img_path):
                    self.serve_file(img_path, "image/png")
                    return
            self.send_error(404)

        else:
            if parsed.path == "/":
                parsed = parsed._replace(path="/index.html")
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), parsed.path.lstrip("/"))
            if os.path.isfile(filepath):
                ext = os.path.splitext(filepath)[1]
                mime = {"html": "text/html", "js": "application/javascript", "css": "text/css", "png": "image/png"}.get(ext.lstrip("."), "application/octet-stream")
                self.serve_file(filepath, mime)
            else:
                self.send_error(404)

    def serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path, mime):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def get_scenes():
    scenes = []
    if not os.path.isdir(SCENES_DIR):
        return {"error": f"scenes directory not found at {SCENES_DIR}"}

    for entry in os.scandir(SCENES_DIR):
        if not entry.is_dir():
            continue
        m = SCENE_PATTERN.match(entry.name)
        if not m:
            continue
        prefix = m.group(1)
        num = int(m.group(2))

        csv_path = os.path.join(entry.path, "metrics.csv")
        if not os.path.isfile(csv_path):
            continue

        occ, prox = None, None
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("metric") == "avg_occlusion":
                    occ = float(row["value"])
                elif row.get("metric") == "avg_proximity":
                    prox = float(row["value"])

        if occ is None or prox is None:
            continue

        has_image = os.path.isfile(os.path.join(entry.path, "viewpoint_1.png"))
        scenes.append({
            "id":        entry.name,  # e.g. "batch1_001"
            "prefix":    prefix,      # e.g. "batch1"
            "num":       num,         # e.g. 1  (for numeric sort)
            "occlusion": occ,
            "proximity": prox,
            "has_image": has_image,
        })

    scenes.sort(key=lambda s: (s["prefix"], s["num"]))
    return scenes


def get_scene_detail(scene_id):
    scene_path = os.path.join(SCENES_DIR, scene_id)
    if not os.path.isdir(scene_path):
        return {"error": f"Scene {scene_id} not found"}

    m = SCENE_PATTERN.match(scene_id)

    metrics = {}
    csv_path = os.path.join(scene_path, "metrics.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("metric") and row.get("value"):
                    try:
                        metrics[row["metric"]] = float(row["value"])
                    except ValueError:
                        metrics[row["metric"]] = row["value"]

    viewpoints = []
    for i in range(1, 20):
        fname = f"viewpoint_{i}.png"
        if os.path.isfile(os.path.join(scene_path, fname)):
            viewpoints.append(f"/api/image/{scene_id}/{fname}")
        else:
            break

    objects = []
    objects_path = os.path.join(scene_path, "objects.csv")
    if os.path.isfile(objects_path):
        with open(objects_path, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("object_name", "").strip()
                if name and name != "plane":
                    objects.append(name)

    return {
        "id":        scene_id,
        "prefix":    m.group(1) if m else "",
        "num":       int(m.group(2)) if m else 0,
        "metrics":   metrics,
        "viewpoints": viewpoints,
        "objects":   objects,
    }


if __name__ == "__main__":
    print(f"Serving at http://localhost:{PORT}")
    print(f"Reading scenes from: {SCENES_DIR}")
    HTTPServer(("", PORT), Handler).serve_forever()