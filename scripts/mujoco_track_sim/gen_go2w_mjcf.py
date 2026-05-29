# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.expanduser(
    "~/robot_lab/source/robot_lab/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf"
)
URDF_MESH_DIR = os.path.expanduser(
    "~/robot_lab/source/robot_lab/data/Robots/unitree/go2w_description/meshes"
)
OUT_DIR = os.path.join(HERE, "assets")
OUT_MESH_DIR = os.path.join(OUT_DIR, "meshes")
OUT_XML = os.path.join(OUT_DIR, "go2w.xml")

LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINTS = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
DEFAULT_JOINT_POS = {"hip": 0.0, "thigh": 0.8, "calf": -1.5, "foot": 0.0}
LEG_KP = 70.0
LEG_KD = 10.0
WHEEL_KV = 0.5
EFFORT_LIMIT = 23.5
BASE_INIT_HEIGHT = 0.45
GROUND_FRICTION = 0.7


def rpy_to_quat(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y_, z])


def parse_origin(elem):
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if elem is not None:
        o = elem.find("origin")
        if o is not None:
            if o.get("xyz"):
                xyz = np.array([float(v) for v in o.get("xyz").split()])
            if o.get("rpy"):
                rpy = np.array([float(v) for v in o.get("rpy").split()])
    return xyz, rpy


def fmt(arr):
    return " ".join(f"{float(v):.8g}" for v in np.atleast_1d(arr))


def default_q_for(joint_name):
    for key, val in DEFAULT_JOINT_POS.items():
        if key in joint_name:
            return val
    return 0.0


def convert_meshes(urdf_root):
    import trimesh

    os.makedirs(OUT_MESH_DIR, exist_ok=True)
    assets = {}
    for mesh in urdf_root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        base = os.path.basename(fn)
        stem, ext = os.path.splitext(base)
        if stem in assets:
            continue
        src = os.path.join(URDF_MESH_DIR, base)
        if ext.lower() == ".stl":
            shutil.copy(src, os.path.join(OUT_MESH_DIR, base))
            assets[stem] = base
        else:
            m = trimesh.load(src, force="mesh")
            out = stem + ".stl"
            m.export(os.path.join(OUT_MESH_DIR, out))
            assets[stem] = out
    return assets


def add_collision_geoms(body_el, link):
    for col in link.findall("collision"):
        xyz, rpy = parse_origin(col)
        geom = col.find("geometry")
        if geom is None:
            continue
        g = ET.SubElement(body_el, "geom")
        g.set("class", "collision")
        g.set("pos", fmt(xyz))
        g.set("quat", fmt(rpy_to_quat(rpy)))
        box = geom.find("box")
        sph = geom.find("sphere")
        cyl = geom.find("cylinder")
        if box is not None:
            g.set("type", "box")
            g.set("size", fmt(np.array([float(v) for v in box.get("size").split()]) / 2.0))
        elif sph is not None:
            g.set("type", "sphere")
            g.set("size", fmt([float(sph.get("radius"))]))
        elif cyl is not None:
            g.set("type", "cylinder")
            g.set("size", fmt([float(cyl.get("radius")), float(cyl.get("length")) / 2.0]))
        else:
            body_el.remove(g)


def add_visual_geoms(body_el, link, mesh_assets):
    for vis in link.findall("visual"):
        xyz, rpy = parse_origin(vis)
        geom = vis.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        stem = os.path.splitext(os.path.basename(mesh.get("filename")))[0]
        if stem not in mesh_assets:
            continue
        g = ET.SubElement(body_el, "geom")
        g.set("class", "visual")
        g.set("pos", fmt(xyz))
        g.set("quat", fmt(rpy_to_quat(rpy)))
        g.set("type", "mesh")
        g.set("mesh", stem)


def add_inertial(body_el, link):
    inert = link.find("inertial")
    if inert is None:
        return
    mass = float(inert.find("mass").get("value"))
    if mass <= 0:
        return
    xyz, rpy = parse_origin(inert)
    i = inert.find("inertia")
    ixx = float(i.get("ixx")); iyy = float(i.get("iyy")); izz = float(i.get("izz"))
    ixy = float(i.get("ixy")); ixz = float(i.get("ixz")); iyz = float(i.get("iyz"))
    el = ET.SubElement(body_el, "inertial")
    el.set("pos", fmt(xyz))
    if np.any(np.abs(rpy) > 1e-9):
        el.set("quat", fmt(rpy_to_quat(rpy)))
    el.set("mass", f"{mass:.8g}")
    el.set("fullinertia", fmt([ixx, iyy, izz, ixy, ixz, iyz]))


def build():
    urdf = ET.parse(URDF_PATH).getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    joints = urdf.findall("joint")

    children = {}
    child_links = set()
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
        child_links.add(j.find("child").get("link"))
    root_link = [n for n in links if n not in child_links]
    assert len(root_link) == 1, f"expected 1 root link, got {root_link}"
    root_link = root_link[0]

    mesh_assets = convert_meshes(urdf)

    mjcf = ET.Element("mujoco", model="go2w")
    ET.SubElement(mjcf, "compiler", angle="radian", autolimits="true", meshdir="meshes")
    ET.SubElement(mjcf, "option", timestep="0.005", integrator="implicitfast")

    default = ET.SubElement(mjcf, "default")
    dcol = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(dcol, "geom", group="3", contype="1", conaffinity="1", condim="3",
                  friction=f"{GROUND_FRICTION} 0.005 0.0001", rgba="0.5 0.6 0.7 0.4")
    dvis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(dvis, "geom", group="2", contype="0", conaffinity="0", density="0")

    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="512")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.15 0.2", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="4 4", reflectance="0.1")
    for stem, fn in sorted(mesh_assets.items()):
        ET.SubElement(asset, "mesh", name=stem, file=fn)

    world = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  material="grid", condim="3", friction=f"{GROUND_FRICTION} 0.005 0.0001")

    def emit(link_name, parent_el, joint=None):
        link = links[link_name]
        body = ET.SubElement(parent_el, "body", name=link_name)
        if joint is not None:
            jxyz, jrpy = parse_origin(joint)
            body.set("pos", fmt(jxyz))
            body.set("quat", fmt(rpy_to_quat(jrpy)))
            jtype = joint.get("type")
            jname = joint.get("name")
            if jtype in ("revolute", "continuous", "prismatic"):
                axis = joint.find("axis")
                ax = [float(v) for v in axis.get("xyz").split()] if axis is not None else [0, 0, 1]
                jel = ET.SubElement(body, "joint", name=jname, axis=fmt(ax))
                jel.set("type", "slide" if jtype == "prismatic" else "hinge")
                if jtype == "revolute":
                    lim = joint.find("limit")
                    if lim is not None and lim.get("lower") is not None:
                        jel.set("range", f"{lim.get('lower')} {lim.get('upper')}")
                if jname in LEG_JOINTS:
                    jel.set("damping", str(LEG_KD))
                    jel.set("armature", "0.01")
                elif jname in WHEEL_JOINTS:
                    jel.set("armature", "0.01")
        else:
            body.set("pos", f"0 0 {BASE_INIT_HEIGHT}")
            ET.SubElement(body, "freejoint", name="root")

        add_inertial(body, link)
        add_collision_geoms(body, link)
        add_visual_geoms(body, link, mesh_assets)
        if link_name == root_link:
            ET.SubElement(body, "site", name="imu", pos="0 0 0", size="0.01")
        for j in children.get(link_name, []):
            emit(j.find("child").get("link"), body, j)

    emit(root_link, world)

    act = ET.SubElement(mjcf, "actuator")
    for jn in LEG_JOINTS:
        ET.SubElement(act, "position", name=f"act_{jn}", joint=jn, kp=str(LEG_KP),
                      forcerange=f"{-EFFORT_LIMIT} {EFFORT_LIMIT}")
    for jn in WHEEL_JOINTS:
        ET.SubElement(act, "velocity", name=f"act_{jn}", joint=jn, kv=str(WHEEL_KV),
                      forcerange=f"{-EFFORT_LIMIT} {EFFORT_LIMIT}")

    sens = ET.SubElement(mjcf, "sensor")
    ET.SubElement(sens, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sens, "velocimeter", name="imu_vel", site="imu")
    ET.SubElement(sens, "framequat", name="base_quat", objtype="site", objname="imu")
    ET.SubElement(sens, "framepos", name="base_pos", objtype="site", objname="imu")

    qpos = [0, 0, BASE_INIT_HEIGHT, 1, 0, 0, 0]
    for jn in LEG_JOINTS:
        qpos.append(default_q_for(jn))
    for jn in WHEEL_JOINTS:
        qpos.append(0.0)
    kf = ET.SubElement(mjcf, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=fmt(qpos))

    os.makedirs(OUT_DIR, exist_ok=True)
    xml_str = minidom.parseString(ET.tostring(mjcf)).toprettyxml(indent="  ")
    with open(OUT_XML, "w") as f:
        f.write(xml_str)
    fix_keyframe(OUT_XML)
    print(f"wrote {OUT_XML}")
    print(f"  root link: {root_link}  leg joints: {len(LEG_JOINTS)}  wheel joints: {len(WHEEL_JOINTS)}")
    print(f"  meshes: {sorted(mesh_assets.values())}")
    return OUT_XML


def fix_keyframe(xml_path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = BASE_INIT_HEIGHT
    data.qpos[3:7] = [1, 0, 0, 0]
    for jn in LEG_JOINTS + WHEEL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = default_q_for(jn)
    correct_qpos = " ".join(f"{v:.8g}" for v in data.qpos)
    tree = ET.parse(xml_path)
    tree.getroot().find("keyframe").find("key").set("qpos", correct_qpos)
    tree.write(xml_path)


def validate(xml_path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print("\n=== MuJoCo validation ===")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"  total mass: {mujoco.mj_getTotalmass(model):.3f} kg")
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print(f"  actuators ({model.nu}): {act_names}")
    print("  validation OK")


if __name__ == "__main__":
    validate(build())
