"""
Cube stacking demo - SSH compatible with third-person video recording.

Usage (local con ventana):
    MUJOCO_GL=glfw python cube_stack_ssh.py

Usage (headless SSH con GPU):
    MUJOCO_GL=egl python cube_stack_ssh.py
"""

import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

# ── Config ────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
SCENE_XML = str(HERE / "stretch_mujoco/models/scene_cubes.xml")
VIDEO_OUT = str(HERE / "cube_stack_demo.mp4")
HEADLESS  = True          # False para ver ventana MuJoCo (requiere pantalla)
VIDEO_FPS = 10
VIDEO_W, VIDEO_H = 640, 480

CAMERAS = [
    StretchCameras.cam_d435i_rgb,
    StretchCameras.cam_d435i_depth,
    StretchCameras.cam_d405_rgb,
    StretchCameras.cam_overhead,
]

# Posiciones de los cubos en la escena (de scene_cubes.xml, tras caer sobre la mesa)
# Mesa top z ≈ 0.48, cubos mitad-alto = 0.04 → centro en z ≈ 0.52
CUBE_BLUE_XYZ = np.array([-0.04, -0.55, 0.52])
CUBE_RED_XYZ  = np.array([ 0.12, -0.55, 0.52])

GRIPPER_OPEN  =  0.04
GRIPPER_CLOSE = -0.015
ARM_PROBE_DIST = 0.12   # metros para probar dirección del brazo

# ── Utilidades ────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def move(sim, actuator, pos, timeout=8.0):
    sim.move_to(actuator, pos)
    sim.wait_while_is_moving(actuator, timeout=timeout)

def ee_pos(sim):
    """Posición del end-effector en frame mundo (x, y, z)."""
    return sim.get_ee_pose()[:3, 3].copy()

# ── Calibración automática de geometría del brazo ────────────────────────────

def probe_arm_geometry(sim):
    """
    Mueve el brazo un poco y mide en qué dirección se mueve el gripper.
    Devuelve (arm_dir_xy, ee_z_at_lift0, lift_z_ratio).
    """
    log("Calibrando geometría del brazo...")

    # Llevar a posición de referencia limpia
    move(sim, Actuators.lift, 0.5, timeout=10)
    move(sim, Actuators.arm,  0.0, timeout=10)
    time.sleep(0.5)

    ee0 = ee_pos(sim)
    log(f"  EE en lift=0.5 arm=0: {np.round(ee0, 3)}")

    # Probar lift: subir 0.1 m
    move(sim, Actuators.lift, 0.6, timeout=6)
    ee_lift = ee_pos(sim)
    dz_per_lift = (ee_lift[2] - ee0[2]) / 0.1   # ≈ 1.0
    move(sim, Actuators.lift, 0.5, timeout=6)
    log(f"  dz/d_lift ≈ {dz_per_lift:.2f}")

    # Probar brazo: extender ARM_PROBE_DIST
    move(sim, Actuators.arm, ARM_PROBE_DIST, timeout=8)
    time.sleep(0.4)
    ee_arm = ee_pos(sim)
    d_arm = ee_arm[:2] - ee0[:2]          # delta xy al extender
    arm_dir = d_arm / (np.linalg.norm(d_arm) + 1e-9)
    move(sim, Actuators.arm, 0.0, timeout=8)
    time.sleep(0.3)

    log(f"  Dirección del brazo (xy): {np.round(arm_dir, 3)}")
    log(f"  EE xy origen: {np.round(ee0[:2], 3)}")

    # z del EE cuando lift=0.5 y arm=0
    ee_ref = ee_pos(sim)
    return arm_dir, ee_ref, dz_per_lift

# ── Control de base ───────────────────────────────────────────────────────────

def move_base_to(sim, tx, ty, tolerance=0.06, timeout=20.0):
    """Mueve la base al punto (tx, ty) con control proporcional."""
    KP_LIN = 1.5
    KP_ANG = 3.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = sim.pull_status()
        dx = tx - st.base.x
        dy = ty - st.base.y
        dist = np.hypot(dx, dy)
        if dist < tolerance:
            break
        ang_target = np.arctan2(dy, dx)
        ang_err = (ang_target - st.base.theta + np.pi) % (2 * np.pi) - np.pi
        if abs(ang_err) > 0.4:
            sim.set_base_velocity(0, KP_ANG * ang_err)
        else:
            sim.set_base_velocity(min(KP_LIN * dist, 0.25),
                                  KP_ANG * ang_err)
        time.sleep(0.04)
    sim.set_base_velocity(0, 0)
    time.sleep(0.3)
    st = sim.pull_status()
    log(f"  Base final: ({st.base.x:.3f}, {st.base.y:.3f})")

# ── Detección de cubos ────────────────────────────────────────────────────────

BLUE_LO, BLUE_HI = np.array([100,100,50]), np.array([130,255,255])
RED_LO1, RED_HI1 = np.array([0,120,50]),   np.array([10,255,255])
RED_LO2, RED_HI2 = np.array([170,120,50]), np.array([180,255,255])

def find_cube(frame_bgr, color):
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, BLUE_LO, BLUE_HI) if color == "blue"
            else cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                                cv2.inRange(hsv, RED_LO2, RED_HI2)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 50:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

# ── Pick and place ────────────────────────────────────────────────────────────

def pick_and_place(sim, arm_dir, ee_ref, dz_per_lift):
    """
    Usa la geometría calibrada para agarrar el cubo azul
    y colocarlo sobre el cubo rojo.
    """

    def compute_joints(target_xyz, arm_ext_margin=0.0):
        """
        Dado un target en mundo, calcula lift y arm necesarios.
        Asume que la base ya está bien posicionada.
        """
        # Z: lift necesario para que el EE quede a la altura target
        dz = target_xyz[2] - ee_ref[2]
        lift_needed = 0.5 + dz / dz_per_lift
        lift_needed = float(np.clip(lift_needed, 0.05, 1.0))

        # XY: distancia desde EE_ref hasta target en dirección del brazo
        d_xy = target_xyz[:2] - ee_ref[:2]
        arm_needed = float(np.dot(d_xy, arm_dir)) + arm_ext_margin
        arm_needed = float(np.clip(arm_needed, 0.0, 0.52))
        return lift_needed, arm_needed

    # ── 1. Posicionar base para cubo azul ────────────────────────────────────
    log("Fase: posicionando base frente al cubo azul...")
    # La base debe estar en: cube_xy - arm_dir * arm_ext
    # Dejamos arm_ext ≈ 0.35m para tener algo de margen
    desired_arm = 0.35
    base_target = CUBE_BLUE_XYZ[:2] - arm_dir * desired_arm
    move_base_to(sim, base_target[0], base_target[1])

    # ── 2. Apuntar cabeza y detectar ──────────────────────────────────────────
    log("Fase: apuntando cámara a la mesa...")
    move(sim, Actuators.head_tilt, -0.85, timeout=5)
    move(sim, Actuators.head_pan,   0.0,  timeout=5)
    time.sleep(0.4)

    try:
        cam_data = sim.pull_camera_data()
        head_rgb = cam_data.get_camera_data(StretchCameras.cam_d435i_rgb)
        blue_px = find_cube(head_rgb, "blue")
        red_px  = find_cube(head_rgb, "red")
        log(f"  Cubo azul en pixel: {blue_px}  |  Cubo rojo en pixel: {red_px}")
    except Exception as e:
        log(f"  Detección de cámara falló: {e}")

    # ── 3. Preparar muñeca ────────────────────────────────────────────────────
    log("Fase: preparando muñeca...")
    move(sim, Actuators.wrist_pitch, -0.9, timeout=5)
    move(sim, Actuators.wrist_yaw,    0.0, timeout=5)
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)

    # ── 4. Calcular y mover al cubo azul ─────────────────────────────────────
    # Recalcular ee_ref con la nueva posición de base
    move(sim, Actuators.lift, 0.5, timeout=8)
    move(sim, Actuators.arm,  0.0, timeout=8)
    time.sleep(0.3)
    ee_ref_now = ee_pos(sim)

    def compute_joints_now(target_xyz, dz_offset=0.0):
        dz = (target_xyz[2] + dz_offset) - ee_ref_now[2]
        lift = float(np.clip(0.5 + dz / dz_per_lift, 0.05, 1.0))
        d_xy = target_xyz[:2] - ee_ref_now[:2]
        arm  = float(np.clip(np.dot(d_xy, arm_dir), 0.0, 0.50))
        return lift, arm

    log("Fase: moviendo sobre cubo azul...")
    lift_hover, arm_blue = compute_joints_now(CUBE_BLUE_XYZ, dz_offset=0.06)
    log(f"  lift={lift_hover:.3f}  arm={arm_blue:.3f}")
    move(sim, Actuators.lift, lift_hover, timeout=8)
    move(sim, Actuators.arm,  arm_blue,   timeout=8)
    time.sleep(0.4)
    log(f"  EE actual: {np.round(ee_pos(sim), 3)}")

    # ── 5. Bajar y agarrar ────────────────────────────────────────────────────
    log("Fase: bajando al cubo azul...")
    lift_grasp, _ = compute_joints_now(CUBE_BLUE_XYZ, dz_offset=-0.01)
    move(sim, Actuators.lift, lift_grasp, timeout=6)
    time.sleep(0.4)

    log("Fase: cerrando gripper...")
    move(sim, Actuators.gripper, GRIPPER_CLOSE, timeout=4)
    time.sleep(0.5)

    # ── 6. Levantar ───────────────────────────────────────────────────────────
    log("Fase: levantando cubo azul...")
    lift_carry, _ = compute_joints_now(CUBE_BLUE_XYZ, dz_offset=0.20)
    move(sim, Actuators.lift, lift_carry, timeout=8)
    time.sleep(0.5)

    # ── 7. Posicionar sobre cubo rojo ────────────────────────────────────────
    log("Fase: moviéndose al cubo rojo...")

    # Mover base lateralmente si los cubos difieren mucho en x o y
    base_target_red = CUBE_RED_XYZ[:2] - arm_dir * desired_arm
    move_base_to(sim, base_target_red[0], base_target_red[1])

    move(sim, Actuators.lift, 0.5, timeout=6)
    move(sim, Actuators.arm,  0.0, timeout=6)
    time.sleep(0.3)
    ee_ref_now = ee_pos(sim)

    # Hover sobre cubo rojo (encima del cubo rojo + alto del cubo azul)
    lift_over, arm_red = compute_joints_now(CUBE_RED_XYZ, dz_offset=0.10)
    log(f"  lift={lift_over:.3f}  arm={arm_red:.3f}")
    move(sim, Actuators.lift, lift_over, timeout=8)
    move(sim, Actuators.arm,  arm_red,   timeout=8)
    time.sleep(0.4)

    # ── 8. Bajar y soltar ────────────────────────────────────────────────────
    log("Fase: colocando cubo azul sobre rojo...")
    lift_place, _ = compute_joints_now(CUBE_RED_XYZ, dz_offset=0.06)
    move(sim, Actuators.lift, lift_place, timeout=6)
    time.sleep(0.4)

    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)
    time.sleep(0.5)

    # ── 9. Retirar ───────────────────────────────────────────────────────────
    log("Fase: retirando brazo...")
    move(sim, Actuators.lift, lift_carry, timeout=6)
    move(sim, Actuators.arm,  0.0,        timeout=6)
    time.sleep(1.0)
    log("¡Completado!")

# ── Grabación de video ────────────────────────────────────────────────────────

class VideoRecorder:
    def __init__(self, sim, path):
        self.sim = sim
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))
        self.label = ""
        self._running = False

    def _loop(self):
        interval = 1.0 / VIDEO_FPS
        while self._running and self.sim.is_running():
            t0 = time.perf_counter()
            try:
                cam = self.sim.pull_camera_data()
                frame = cam.get_camera_data(StretchCameras.cam_overhead)
                # overlay de detección
                try:
                    head = cam.get_camera_data(StretchCameras.cam_d435i_rgb)
                    bc = find_cube(head, "blue")
                    rc = find_cube(head, "red")
                    if bc: cv2.circle(frame, bc, 15, (255,80,0), 3)
                    if rc: cv2.circle(frame, rc, 15, (0,50,255), 3)
                except Exception:
                    pass
                if self.label:
                    cv2.putText(frame, self.label, (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)
                if frame.shape[:2] != (VIDEO_H, VIDEO_W):
                    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
                self.writer.write(frame)
            except Exception:
                pass
            time.sleep(max(0, interval - (time.perf_counter() - t0)))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        time.sleep(0.3)
        self.writer.release()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Cube Stack Demo")
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL','no seteado')}")
    print(f"  Escena    = {SCENE_XML}")
    print(f"  Video     = {VIDEO_OUT}")
    print("=" * 60)

    sim = StretchMujocoSimulator(
        scene_xml_path=SCENE_XML,
        cameras_to_use=CAMERAS,
        camera_hz=10,
    )

    log("Iniciando simulación...")
    sim.start(headless=HEADLESS)

    if not sim.is_running():
        log("ERROR: la simulación no arrancó.")
        sys.exit(1)

    rec = VideoRecorder(sim, VIDEO_OUT)
    rec.start()
    log(f"Grabando en: {VIDEO_OUT}")

    try:
        # Esperar que los cubos caigan y se estabilicen
        rec.label = "Esperando estabilización..."
        log("Esperando que los cubos se estabilicen...")
        time.sleep(3.0)

        # Calibrar geometría del brazo
        rec.label = "Calibrando brazo..."
        arm_dir, ee_ref, dz_per_lift = probe_arm_geometry(sim)

        # Ejecutar pick-and-place
        rec.label = "Pick and place..."
        pick_and_place(sim, arm_dir, ee_ref, dz_per_lift)

        rec.label = "Completado"
        time.sleep(3.0)

    except KeyboardInterrupt:
        log("Interrumpido.")
    finally:
        rec.stop()
        log(f"Video guardado: {VIDEO_OUT}")
        sim.stop()

if __name__ == "__main__":
    main()
