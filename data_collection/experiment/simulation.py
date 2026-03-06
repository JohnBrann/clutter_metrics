from pathlib import Path
import time

import numpy as np
import pybullet

from utils.grasp import Label
from utils.perception import *
from experiment import btsim, workspace_lines
from utils.transform import Rotation, Transform
from utils.noise import apply_noise
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Slerp
import json


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, save_dir=None, save_freq=8, remove_box=True, replica_scene_id=0):
        assert scene in ["pile", "packed", "replica", "shelf"]

        self.urdf_root = Path("object_sets")
        self.scene = scene
        self.object_set = object_set
        self.replica_scene_id = replica_scene_id
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        # self.gripper = Gripper(self.world)
        # self.size = 6 * self.gripper.finger_depth
        self.size = 6 * 0.05
        if self.scene == "shelf":
            self.size = 0.5
        # intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.intrinsic = CameraIntrinsic(848, 480, 426.678, 426.67822265625, 427.2525634765625, 234.44296264648438)
        self.camera = self.world.add_camera(self.intrinsic, 0.1, 2.0)

        self.remove_box = remove_box

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    # def discover_objects(self):
    #     root = self.urdf_root / self.object_set
    #     self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

     # Discovers objects in the structure expected from MOAD datasets
    def discover_objects(self):
        root = self.urdf_root / self.object_set
        print(f'root: {root}')
        self.object_urdfs = sorted(
            (p for p in root.glob("*/fused/*.urdf") if p.is_file()),
            key=lambda p: p.stem
        )
        print(f'Discoverted Objects: {self.object_urdfs}')

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        # self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = 0.01
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        elif self.scene == "replica":
            replica_scene_path = Path("scene_replica_scenes") / "scenes"
            self.generate_replica_scene(table_height, self.replica_scene_id, replica_scene_path)
        elif self.scene == "shelf":
            self.generate_shelf_scene(object_count, table_height)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        nlz = 0.005
        self.lower = np.r_[lx, ly, lz]
        self.newlower = np.r_[lx, ly, nlz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            # print(f'{urdf} xy: {xy}')
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(1.0, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=3.0)

        # remove box
        if self.remove_box == True:
            self.world.remove_body(box)
            self.remove_and_wait()

        self.wait_for_objects_to_rest(timeout=3.0)
        
        # save pos and quaternion of each object to a json file
        self.save_poses_to_json()
        self.save_poses_to_npz()


    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()

           
            # self.remove_and_wait()

            attempts += 1

        self.wait_for_objects_to_rest(timeout=3.0)
        
        # positions = [
        #     {'x': 0.3, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.3, 'y': 0.3, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0, 'y': 0.3, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.15, 'y': 0.15, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.05, 'y': 0.05, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.25, 'y': 0.25, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.05, 'y': 0.15, 'angle': 0.0, 'scale': 1.0},
        # ]

        # # positions = [
        # #     {'x': 0, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        # # ]
        # test_urdf = self.object_urdfs[0]

        # for pos in positions:
        #     urdf = test_urdf
        #     x, y = pos['x'], pos['y']
        #     angle = pos.get('angle', 0.0)
        #     scale = pos.get('scale', 1.0)
            
        #     z = 1.0
        #     # rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
        #     rotation = Rotation.random(random_state=self.rng)
        #     pose = Transform(rotation, np.r_[x, y, z])
        #     body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            
        #     lower, upper = self.world.p.getAABB(body.uid)
        #     z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
        #     body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
        #     self.world.step()
        #     self.wait_for_objects_to_rest(timeout=3.0)

        # save pos and quaternion of each object to a json file
        self.save_poses_to_json()
        self.save_poses_to_npz()


    def generate_replica_scene(self, table_height, scene_number, replica_scene_path):
        # Get scene data
        full_scene_path = Path(replica_scene_path) / f"{scene_number}.npz"
        data = np.load(full_scene_path, allow_pickle=True)

        model_names = data["model_names"] 
        print(f'Model Names: {model_names}')
        # EX. Model Names: ['040_large_marker' '010_potted_meat_can' '006_mustard_bottle' '004_sugar_box' '035_power_drill']

        poses = data["poses"]  # [x, y, z, qx, qy, qz, qw]
        print(f'Poses: {poses}')
        
        # Extracts the end of each model name path so we can use it to match the object names in the .npz file
        urdf_by_name = {p.stem: p for p in self.object_urdfs}
        print(f'urdfs : {urdf_by_name}')

        # Place objects
        for model_name, pose7 in zip(model_names, poses):
            urdf = urdf_by_name[model_name]
            print(f'urdf: {urdf}')
            # EX. urdf: object_sets/ycb/040_large_marker/fused/040_large_marker.urdf
            
            x_, y_, z, qx, qy, qz, qw = pose7
            x = x_ + 0.15
            y = y_ + 0.15
            rot = Rotation.from_quat([qx, qy, qz, qw])

            # Load object
            pose = Transform(rot, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=1.3)

            # Adjust z so object sits on the table
            lower, upper = self.world.p.getAABB(body.uid)
            z_on_table = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rot, np.r_[x, y, z_on_table]))

            self.world.step()
            self.wait_for_objects_to_rest(timeout=3.0)

        self.wait_for_objects_to_rest(timeout=3.0)
       

    def generate_shelf_scene(self, object_count, table_height):
        urdf = self.urdf_root / "setup" / "shelf4.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.21, 0.21, table_height])
        shelf1 = self.world.load_urdf(urdf, pose, scale=1.3)

        # manually placeing objects on shelf
        positions = [
            {'x': 0.1, 'y': 0.1, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 1},
            {'x': 0.3, 'y': 0.1, 'z': 0.1, 'angle': 0.0, 'scale': 1.0, 'urdf': 10},
            {'x': 0.1, 'y': 0.25, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 14},
            {'x': 0.25, 'y': 0.2, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 4},
            {'x': 0.25, 'y': 0.2, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 4},
            {'x': 0.25, 'y': 0.22, 'z': 0.1, 'angle': 0.0, 'scale': 1.0, 'urdf': 4},
            {'x': 0.35, 'y': 0.2, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 4},
            {'x': 0.35, 'y': 0.12, 'z': 0.3, 'angle': 0.0, 'scale': 1.0, 'urdf': 1},
            {'x': 0.1, 'y': 0.1, 'z': 0.1, 'angle': 0.0, 'scale': 1.0, 'urdf': 3},
            {'x': 0.1, 'y': 0.2, 'z': 0.1, 'angle': 0.0, 'scale': 1.0, 'urdf': 7},
            {'x': 0.3, 'y': 0.25, 'z': 0.1, 'angle': 0.0, 'scale': 1.0, 'urdf': 14},
        ]

        for pos in positions:
            urdf = self.object_urdfs[pos['urdf']]
            x, y, z = pos['x'], pos['y'], pos['z']
            angle = pos.get('angle', 0.0)
            scale = pos.get('scale', 1.0)
            
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            # rotation = Rotation.random(random_state=self.rng)
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            
            lower, upper = self.world.p.getAABB(body.uid)
            # z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()
            self.wait_for_objects_to_rest(timeout=3.0)

        # save pos and quaternion of each object to a json file
        self.save_poses_to_json()
        self.save_poses_to_npz()


    def save_poses_to_npz(self, path="scene_poses.npz"):
        """
        Save current world object poses to NPZ.
        Stores:
        - ids:        (N,)   int
        - poses:      (N,7)  float  [x,y,z,qx,qy,qz,qw]  (PyBullet quat is xyzw)
        - model_names:(N,)   str    (should match urdf_by_name keys, e.g. urdf Path.stem)
        """
        ids = []
        poses = []
        model_names = []

        for uid, body in list(self.world.bodies.items()):
            
            uid_i = int(uid)
            if uid_i == 0:
                print("[info] skipping plane (uid=0)")
                continue
            pos, orn = self.world.p.getBasePositionAndOrientation(uid_i)

            name = getattr(body, "name", "") or ""
            name_stem = Path(name).stem if name else ""
         
            ids.append(uid_i)
            poses.append([
                float(pos[0]-0.15), float(pos[1]-0.15), float(pos[2]),
                float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])
            ])
            model_names.append(name_stem)

            # print(f"[info] saved uid={uid_i} name='{name_stem}' "
            #     f"pos={[float(pos[0]), float(pos[1]), float(pos[2])]} "
            #     f"quat={[float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])]}")
        

        ids = np.asarray(ids, dtype=np.int32)
        poses = np.asarray(poses, dtype=np.float32)

        # Store as unicode strings (best for npz). If you truly need arbitrary python
        # objects, you can store dtype=object instead.
        model_names = np.asarray(model_names, dtype=np.str_)

        np.savez(path, ids=ids, poses=poses, model_names=model_names)
        print(f"[info] scene saved to {path}")

    
    def save_poses_to_json(self, path="scene_poses.json"):
        """
        Save current world object poses to JSON.
        Each item contains: uid (int), name (str), pos [x,y,z], orn [x,y,z,w], optional scale.
        """
        out = {
            "objects": []
        }

        for uid, body in list(self.world.bodies.items()):
            try:
                uid_i = int(uid)
                pos, orn = self.world.p.getBasePositionAndOrientation(uid_i)
                name = getattr(body, "name", "") or ""
                # include any known scale if your bodies carry that info, else default 1.0
                scale = getattr(body, "scale", None)
                item = {
                    "uid": uid_i,
                    "name": name,
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    # PyBullet ordering: (x, y, z, w)
                    "quat_xyzw": [float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])]
                }
                if scale is not None:
                    item["scale"] = float(scale)
                out["objects"].append(item)

                print(f"[info] saved uid={uid_i} name='{name}' pos={item['pos']} quat={item['quat_xyzw']}")
            except Exception as e:
                print(f"[warn] failed to read pose for uid={uid}: {e}")

        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[info] scene saved to {path}")




    def recovered_scene(self, mesh_list):
        # texture_id = world.p.loadTexture('/home/pinhao/Desktop/GIGA/texture_0.jpg')
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        # self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)
        for (mesh_path, scale, pose) in mesh_list:
            pose = Transform.from_matrix(pose)
            mesh_path = '_'.join(mesh_path.split('_')[:-1])+'.urdf'
            body = self.world.load_urdf(mesh_path, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            # body.set_pose(pose=pose)
            self.world.step()

    def advance_sim(self,frames):
        for _ in range(frames):
            self.world.step()

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object
    

    def idle(self, hz=240):
        """Keep the sim running without executing any grasps."""
        import time
        print("Scene spawned. Press Ctrl+C to exit.")
        try:
            while True:
                self.world.step()          # uses btsim.BtWorld.step()
                time.sleep(1.0 / hz)       # gentle throttle
        except KeyboardInterrupt:
            pass


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("object_sets/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )
    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid!=self.body.uid:
                return grased_id.uid
            
    def get_distance_from_hand(self,):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand
    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
    
    def move_gripper_top_down(self):
        current_pose = self.body.get_pose()
        pos = current_pose.translation + 0.1
        flip = Rotation.from_euler('y', np.pi)
        target_ori = Rotation.identity()*flip
        self.move_tcp_pose(Transform(rotation=target_ori,translation=pos),abs=True)
    
    def move_tcp_pose(self, target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pos_diff = target.translation - T_world_tcp.translation
        n_steps = max(int(np.linalg.norm(pos_diff) / eef_step1),10)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1
        key_rots = np.stack((T_world_body.rotation.as_quat(),target.rotation.as_quat()),axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0,1.0],key_rots)
        times = np.linspace(0,1,n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_tcp.translation += dist_step
            T_world_tcp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.constraint.change(
                    jointChildPivot=T_world_tcp.translation,
                    jointChildFrameOrientation=T_world_tcp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
    
    def shake_hand(self,pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()
        x,y,z = current_pose.translation[0],current_pose.translation[1],current_pose.translation[2]
        default_position = [x, y, z]
        shake_position = [x, y, z+0.05]
        hand_orientation2 = pybullet.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
        shake_orientation1 = pybullet.getQuaternionFromEuler([np.pi, -np.pi / 12, -np.pi/2])
        shake_orientation2 = pybullet.getQuaternionFromEuler([np.pi, np.pi / 12, -np.pi/2])
        new_trans = current_pose.translation + np.array([0.,0.,0.05])
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2),translation=new_trans))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=shake_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        else:
            return True
        
    def is_dropped(self,object_id,prev_dist):
        pos,_ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist,dist_from_hand,atol=0.1):
            return False
        else:
            return True