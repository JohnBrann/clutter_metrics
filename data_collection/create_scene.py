#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import json
import numpy as np
from uuid import uuid4
import math
import yaml
import random
import csv
import os

from utils.perception import camera_on_sphere
from experiment.simulation import ClutterRemovalSim
from utils.transform import Rotation, Transform
from utils.io import write_setup 

R_FACTOR =2.0                      
PHI = -math.pi / 2                  
BASE_PHI_RAD = -math.pi / 2
EPS_THETA = 1e-3                  
THETA_STEP_DEG = 18
AZIMUTH_STEP_DEG = 10
VIEW_THETAS_DEG = sorted(set(list(range(0, 90, THETA_STEP_DEG)) + [90]))

# Generate azimuth offsets (0..359 by step). 360 is equivalent to 0 so we stop at 360 (exclusive).
AZIMUTH_OFFSETS_DEG = list(range(0, 360, AZIMUTH_STEP_DEG))


# ----------------------------
# GUI camera helper
# ----------------------------
def set_gui_camera_from_sphere(sim, r, theta, phi):
    """Match the offscreen camera with the PyBullet GUI camera so you can watch it."""
    sim.world.p.resetDebugVisualizerCamera(
        cameraDistance=float(r),
        cameraYaw=float(np.degrees(phi)),
        cameraPitch=float(-np.degrees(theta)),
        cameraTargetPosition=[float(sim.size/2), float(sim.size/2), 0.0],
    )

def extrinsics_to_spherical(T, center, scene):
    """
    Given camera Transform T and scene center, return (r, theta, phi)
    with theta in [0, pi], phi in (-pi, pi], and their degree versions.
    """
    vec = T.translation - center
    if scene == "shelf":
        r = 2.5
    else:
        r = 0.5
        # r = max(np.linalg.norm(vec), 1e-9)

    theta = math.acos(max(min(vec[2]/r, 1.0), -1.0))  # polar angle from +Z
    phi = math.atan2(vec[1], vec[0])
    return r, theta, phi, np.degrees(theta), np.degrees(phi)

# ----------------------------
# Color LUT generation
# ----------------------------
def generate_uid_color_lut(obj_uids):
    """
    Generate a stable color lookup table for object UIDs.
    Uses a deterministic hash-based color generation for consistency.
    Returns: (K, 4) array with RGBA values (0-255), where A=255.
    """
    K = len(obj_uids)
    colors = np.zeros((K, 4), dtype=np.uint8)
    
    for i, uid in enumerate(obj_uids):
        # Use UID as seed for deterministic color generation
        rng = np.random.RandomState(int(uid) * 12345)
        # Generate vibrant colors (avoid very dark or very light)
        rgb = rng.randint(50, 230, size=3, dtype=np.uint8)
        colors[i] = [rgb[0], rgb[1], rgb[2], 255]
    
    return colors

# ----------------------------
# Build extrinsics 
# ----------------------------
# def build_extrinsics(sim):
#     """
#     Use VIEW_THETAS_DEG, R_FACTOR, PHI to create camera extrinsics.
#     """
#     thetas = [max(math.radians(t), EPS_THETA) for t in VIEW_THETAS_DEG]  # clamp away from 0
#     origin = Transform(Rotation.identity(), np.r_[sim.size/2, sim.size/2, sim.size/3])
#     r = R_FACTOR * sim.size
#     extrinsics = [camera_on_sphere(origin, r, th, PHI) for th in thetas]
#     print(f"[info] Viewpoints (deg): {VIEW_THETAS_DEG} (0° nudged by {EPS_THETA} rad), phi={PHI:.3f}, r≈{r:.3f}")
#     return extrinsics

def build_extrinsics(sim):
    """
    Create camera extrinsics for each azimuth offset * each elevation.
    Returns: (extrinsics_list, theta_phi_pairs)
    """
    thetas = [max(math.radians(t), EPS_THETA) for t in VIEW_THETAS_DEG]
    phis   = [BASE_PHI_RAD + math.radians(d) for d in AZIMUTH_OFFSETS_DEG]

    # origin = Transform(Rotation.identity(), np.r_[sim.size/2, sim.size/2, sim.size/3])
    origin = Transform(Rotation.identity(), np.r_[0.0, 0.0, 0.0])
    r = R_FACTOR * sim.size

    extrinsics = []
    theta_phi_pairs = []  # Store the input angles
    
    for ph_deg, ph in zip(AZIMUTH_OFFSETS_DEG, phis):
        for th_deg, th in zip(VIEW_THETAS_DEG, thetas):
            extrinsics.append(camera_on_sphere(origin, r, th, ph))
            theta_phi_pairs.append((th_deg, ph_deg))  # Store input degrees directly

    # print(f"R: {r}")
    # print(f"Theta, Phi:\n{theta_phi_pairs}")


    # print(f"[info] Viewpoints: {len(phis)} azimuth(s) x {len(thetas)} elevation(s) "
    #       f"= {len(extrinsics)} total")
    # print(f"[info] Azimuths deg: {AZIMUTH_OFFSETS_DEG} (base={math.degrees(BASE_PHI_RAD):.1f}°)")
    # print(f"[info] Elevations deg: {VIEW_THETAS_DEG} (0° nudged by {EPS_THETA} rad), r≈{r:.3f}")
    
    return extrinsics, theta_phi_pairs

# ----------------------------
# Saving helpers
# ----------------------------
def save_scene_single_file(out_dir: Path,
                           scene_id: str,
                           scene_name: str,
                           depth_imgs: np.ndarray,
                           seg_imgs: np.ndarray,
                           extr_list: np.ndarray,
                           per_obj_masks: np.ndarray,
                           per_obj_seg_uids: np.ndarray,
                           uid_color_lut: np.ndarray,
                           obj_uids: np.ndarray,
                           bodies_json: str,
                           view_theta_deg: np.ndarray,
                           view_phi_deg: np.ndarray):
    """
    Save EVERYTHING for this scene into one NPZ:
      - depth_imgs (N,H,W) float32
      - seg_imgs   (N,H,W) uint16
      - extrinsics (N,7)   float32
      - per_obj_masks (N,K,H,W) uint8 - binary masks
      - per_obj_seg_uids (N,K,H,W) int32 - UID-labeled masks
      - uid_color_lut (K,4) uint8 - RGBA colors for each UID
      - obj_uids   (K,)    int32
      - bodies_json (str)
      - view_theta_deg (N,) float32
      - view_phi_deg   (N,) float32
      - scene_id (str)
    """
    # scenes_dir = save_root / "scenes"
    # scenes_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_scene_data.npz"
    np.savez_compressed(
        out_path,
        scene_id=np.array(scene_id),
        depth_imgs=depth_imgs,
        seg_imgs=seg_imgs,
        extrinsics=extr_list,
        per_obj_masks=per_obj_masks,
        per_obj_seg_uids=per_obj_seg_uids,
        uid_color_lut=uid_color_lut,
        obj_uids=np.asarray(obj_uids, dtype=np.int32),
        bodies_json=np.array(bodies_json),
        view_theta_deg=view_theta_deg.astype(np.float32),
        view_phi_deg=view_phi_deg.astype(np.float32),
    )
    print(f"[saved] single-file scene -> {out_path.resolve()}")
    return out_path

# ----------------------------
# Object discovery (verbose)
# ----------------------------
def discover_scene_bodies_verbose(world):
    """
    Return:
      - bodies: list of dicts {uid, name}
      - object_uids: filtered list of UIDs we consider 'objects'
    """
    bodies = []
    for uid, body in world.bodies.items():
        name = getattr(body, "name", "")
        bodies.append({"uid": int(uid), "name": str(name)})

    # Filter rules: exclude common non-targets by name (lowercased)
    def is_object(uid, name_lc):
        if any(s in name_lc for s in ["plane", "table", "panda", "hand"]):
            return False
        # Exclude the drop "box", but allow "block"
        # if name_lc == "box" or name_lc.endswith("/box.urdf"):
        #     return False
        return True

    object_uids = []
    for entry in bodies:
        uid = entry["uid"]
        name_lc = entry["name"].lower()
        if is_object(uid, name_lc):
            object_uids.append(uid)

    return bodies, object_uids

# ----------------------------
# Segmentation decoding
# ----------------------------
def seg_uid_image(seg_img: np.ndarray) -> np.ndarray:
    """
    Decode objectUniqueId per pixel from PyBullet segmentation image.
    Background may be -1. We zero non-positive to avoid spurious matches.
    Common encoding: seg = objectUid + ((linkIndex+1) << 24)  or seg = (linkIndex << 24) + objectUid
    Using the lower 24 bits for uid works across builds.
    """
    s = seg_img.astype(np.int32)
    uid = s & ((1 << 24) - 1)
    uid[s <= 0] = 0
    return uid

# ----------------------------
# Visibility utils
# ----------------------------
def set_body_visibility(p, body_uid: int, visible: bool):
    """
    Make an entire multibody visible/invisible by changing RGBA alpha of all its visual links.
    No movement of bodies is performed.
    """
    try:
        n_joints = p.getNumJoints(body_uid)
    except Exception:
        return
    alpha = 1.0 if visible else 0.0
    p.changeVisualShape(body_uid, -1, rgbaColor=[1, 1, 1, alpha])  # base link
    for link in range(n_joints):
        p.changeVisualShape(body_uid, link, rgbaColor=[1, 1, 1, alpha])

def set_all_bodies_visibility(world, visible: bool):
    """Set all bodies to the same visibility (no movement)."""
    p = world.p
    for uid in list(world.bodies.keys()):
        set_body_visibility(p, int(uid), visible)

def set_only_target_visible(world, target_uid: int):
    """Hide all bodies except the target (no movement)."""
    p = world.p
    for uid in list(world.bodies.keys()):
        set_body_visibility(p, int(uid), uid == target_uid)

# ----------------------------
# Body pose management for clean solo renders
# ----------------------------
def save_body_poses(p, body_uids):
    """Save position and orientation of all bodies."""
    poses = {}
    for uid in body_uids:
        try:
            pos, orn = p.getBasePositionAndOrientation(int(uid))
            poses[int(uid)] = (pos, orn)
        except Exception:
            pass
    return poses

def restore_body_poses(p, poses):
    """Restore position and orientation of all bodies."""
    for uid, (pos, orn) in poses.items():
        try:
            p.resetBasePositionAndOrientation(int(uid), pos, orn)
        except Exception:
            pass

# ----------------------------
# Rendering helpers
# ----------------------------
def render_full_scene(sim, T):
    """One full-scene offscreen render (depth + seg)."""
    _, depth_img, seg_img = sim.camera.render(T, return_seg=True)
    # return depth_img, seg_img.astype(np.uint16)
    return depth_img, seg_img.astype(np.int32)

def render_object_solo(sim, T, target_uid, all_body_uids, saved_poses):
    p = sim.world.p
    world = sim.world

    # Pause GUI rendering while we modify the scene (avoids flicker)
    if world.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # Move all non-target objects far away (outside camera frustum)
    # This ensures they don't occlude the target in the depth buffer
    far_away = [10000.0, 10000.0, 10000.0]
    for uid in all_body_uids:
        if uid != target_uid and uid in saved_poses:
            try:
                _, orn = saved_poses[uid]  # Keep original orientation
                p.resetBasePositionAndOrientation(int(uid), far_away, orn)
            except Exception:
                pass

    if world.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Render with only the target object in view (full ground truth mask!)
    _, depth_solo, seg_solo = sim.camera.render(T, return_seg=True)
    uid_img = seg_uid_image(seg_solo)
    
    # Binary mask: where this object appears
    mask = (uid_img == int(target_uid)).astype(np.uint8)
    
    # UID-labeled mask: pixels labeled with the object's UID (0 elsewhere)
    uid_mask = np.where(uid_img == int(target_uid), int(target_uid), 0).astype(np.int32)
    
    return mask, uid_mask, depth_solo



def save_viewpoint_images(sim, out_dir: Path, target=(0.0, 0.0, 0.0), elevation_deg=45.0, radius=0.5):
    """
    Capture 4 RGB images around the scene at 45-degree elevation,
    each 90 degrees apart in azimuth, all looking at target.
    Saves as viewpoint_1.png ... viewpoint_4.png in out_dir.
    """
    from PIL import Image

    azimuths_deg = [0, 90, 180, 270]
    elevation_rad = math.radians(elevation_deg)

    for i, az_deg in enumerate(azimuths_deg):
        az_rad = math.radians(az_deg)

        # Compute camera position on sphere around target
        eye_x = target[0] + radius * math.cos(elevation_rad) * math.cos(az_rad)
        eye_y = target[1] + radius * math.cos(elevation_rad) * math.sin(az_rad)
        eye_z = target[2] + radius * math.sin(elevation_rad)

        view_matrix = sim.world.p.computeViewMatrix(
            cameraEyePosition=[eye_x, eye_y, eye_z],
            cameraTargetPosition=list(target),
            cameraUpVector=[0, 0, 1]
        )

        W = sim.camera.intrinsic.width
        H = sim.camera.intrinsic.height

        projection_matrix = sim.world.p.computeProjectionMatrixFOV(
            fov=60,
            aspect=W / H,
            nearVal=0.01,
            farVal=10.0
        )

        _, _, rgb, _, _ = sim.world.p.getCameraImage(
            width=W,
            height=H,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=sim.world.p.ER_TINY_RENDERER
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        out_path = out_dir / f"viewpoint_{i+1}.png"
        Image.fromarray(rgb_array).save(out_path)
        # print(f"[saved] viewpoint_{i+1}.png -> {out_path.resolve()}")


# ----------------------------
# Main (per-viewpoint ordering; viewpoints from in-code list only)
# ----------------------------
def main():

    # Load scene cfg file: config/scene_config.yaml
    with open("config/scene_config.yaml", "r") as file:
        scene_config = yaml.safe_load(file)

    scene_type = scene_config["scene_type"]
    scene_name = scene_config["scene_name"]
    scene_prefix = scene_config["scene_prefix"]

    # Object Parameters
    object_set = scene_config["object_set"]
    number_of_objects = random.randint(scene_config["num_objects"][0], scene_config["num_objects"][1])
    allow_duplicate_objects = scene_config["allow_duplicate_objects"]
    list_of_objects = scene_config["list_of_objects"]

    # Boundary Parameters
    boundary_type = scene_config["boundary_shape"]
    boundary_radius = random.uniform(scene_config["boundary_radius"][0], scene_config["boundary_radius"][1])
    remove_boundary = scene_config["remove_boundary"]
    

    # Place holders for future specific config settings
    print(f'\nScene Configuration:')    

    scene_name = scene_prefix + scene_name
    print(f'Scene Name: {scene_name}')
    print(f'Scene Type: {scene_type}')
    print(f'Object Set: {object_set}')
    print(f'Number of Objects: {number_of_objects}')


    if scene_type == "packed":
        return
    elif scene_type == "replica":
        return


    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between viewpoints.")
    parser.add_argument("--idle-after", action="store_true",
                        help="Idle after the tour so the GUI stays open.")
    parser.add_argument("--replica-scene-id",type=str,
        default="1",help="")

    args = parser.parse_args()

    if scene_type == "replica":
        out_dir = Path("../scenes") / "replica" / args.replica_scene_id
    else:
        out_dir = Path("../scenes") / scene_name


    np.random.seed()
    sim = ClutterRemovalSim(scene_type, object_set, scene_name=scene_name, number_of_objects=number_of_objects, allow_duplicate_objects=allow_duplicate_objects, 
                            list_of_objects=list_of_objects, gui=args.sim_gui, seed=None, save_dir=None, save_freq=8, boundary_type=boundary_type, 
                            boundary_radius=boundary_radius, remove_boundary=remove_boundary, replica_scene_id=1 )

    # Output root + setup metadata
    (out_dir).mkdir(parents=True, exist_ok=True)
    write_setup(
        out_dir,
        sim.size,
        sim.camera.intrinsic,
    )

    # Build one cluttered scene and settle
    object_count = 5
    sim.reset(object_count)

    # Capture RGB images
    save_viewpoint_images(sim, out_dir, target=(0.0, 0.0, 0.0), elevation_deg=45.0, radius=0.4)

    extrinsics, theta_phi_pairs = build_extrinsics(sim)
    extrinsics_xyz = [list(t.translation) for t in extrinsics]
    # print(f'extrinsics:\n {extrinsics_xyz}')


    for i, (th_deg, ph_deg) in enumerate(theta_phi_pairs):
        if ph_deg == 0:
            x, y, z = extrinsics[i].translation
            # print(f"(theta,phi)=({th_deg:03d},{ph_deg:03d}) -> x={x:.6f}, y={y:.6f}, z={z:.6f}")

    # print(f'extrinsics: {extrinsics}')
    center = np.array([sim.size/2, sim.size/2, sim.size/3])

    # Discover bodies/objects once (constant for the scene)
    bodies, obj_uids = discover_scene_bodies_verbose(sim.world)
    print("[info] Bodies in scene:")
    for b in bodies:
        print(f"  - uid={b['uid']:>3}  name='{b['name']}'")
    # print(f"[info] Object UIDs (count={len(obj_uids)}): {obj_uids}")


     # Write to CSV
    objects_csv_path = os.path.join("..", "scenes", scene_name, "objects.csv")
    os.makedirs(os.path.dirname(objects_csv_path), exist_ok=True)
    with open(objects_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "object_name"])
        for b in bodies:
            writer.writerow([b["uid"], b["name"]])
    print(f"[info] Saved objects list to {objects_csv_path}")


    # Generate consistent color LUT for all objects
    uid_color_lut = generate_uid_color_lut(obj_uids)
    # print(f"[info] Generated color LUT for {len(obj_uids)} objects:")
    for i, uid in enumerate(obj_uids):
        rgb = uid_color_lut[i, :3]
        # print(f"  - UID {uid}: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")

    # Allocate outputs
    H, W = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    N = len(extrinsics)
    K = len(obj_uids)

    depth_imgs = np.empty((N, H, W), np.float32)
    seg_imgs   = np.empty((N, H, W), np.uint32)
    extr_list  = np.empty((N, 7), np.float32)
    per_obj_masks = np.zeros((N, K, H, W), dtype=np.uint8)
    per_obj_seg_uids = np.zeros((N, K, H, W), dtype=np.int32)

    view_theta_deg = np.empty((N,), np.float32)
    view_phi_deg   = np.empty((N,), np.float32)

    center = np.array([sim.size/2, sim.size/2, sim.size/3])

    # ---------- Per-VIEWPOINT loop ----------
    for i, T in enumerate(extrinsics):
        # Sync GUI camera and compute angles from the actual transform
        cam_pos = T.translation
        th_deg, ph_deg = theta_phi_pairs[i]
        if sim.gui:
            r, theta, phi, _, _ = extrinsics_to_spherical(T, center, scene_type)
            set_gui_camera_from_sphere(sim, r, theta, phi)
        view_theta_deg[i] = th_deg
        view_phi_deg[i] = ph_deg

        #  render full scene
        set_all_bodies_visibility(sim.world, True)
        d_full, seg_full = render_full_scene(sim, T)
        depth_imgs[i] = d_full
        seg_imgs[i]   = seg_full
        extr_list[i]  = T.to_list()

        # Save body poses once per viewpoint (before any modifications)
        # Get all body UIDs currently in the simulation
        all_body_uids = list(sim.world.bodies.keys())
        saved_poses = save_body_poses(sim.world.p, all_body_uids)

        #  per-object solo renders (removes non-targets for clean masks) ---
        for j, uid in enumerate(obj_uids):
            # Restore all body poses before rendering each object
            restore_body_poses(sim.world.p, saved_poses)
            
            # Remove non-targets and render target alone
            # This gives us the full ground truth mask without occlusion
            mask, uid_mask, _ = render_object_solo(sim, T, uid, all_body_uids, saved_poses)
            per_obj_masks[i, j] = mask
            per_obj_seg_uids[i, j] = uid_mask
            time.sleep(args.delay)

        #  Restore all body poses for next viewpoint
        restore_body_poses(sim.world.p, saved_poses)
        
        if args.delay > 0:
            time.sleep(args.delay)

        # For user-facing GUI between viewpoints, restore visibility
        if sim.gui:
            set_all_bodies_visibility(sim.world, True)

    # ---------- Save single-file artifact ----------
    # scenes_dir = args.root / "scenes"
    # scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_id = uuid4().hex
    bodies_json = json.dumps(bodies)
    out_path = save_scene_single_file(
        out_dir,
        scene_id,
        scene_name,
        depth_imgs,
        seg_imgs,
        extr_list,
        per_obj_masks,
        per_obj_seg_uids,
        uid_color_lut,
        np.array(obj_uids, dtype=np.int32),
        bodies_json,
        view_theta_deg,
        view_phi_deg,
    )

    # Optionally keep GUI running
    if args.sim_gui and args.idle_after:
        print("Finished viewpoint tour. Idling… (Ctrl+C to exit)")
        try:
            while True:
                sim.world.step()
                time.sleep(1/240)
        except KeyboardInterrupt:
            pass

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    main()