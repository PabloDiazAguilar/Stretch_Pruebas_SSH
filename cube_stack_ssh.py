"""
Cube stacking demo — vision-based con ArUco del brazo.

Flujo:
  1. point_head_at(posición estimada) → cabeza mira al brazo
  2. Detecta ArUco de muñeca (ID=133, DICT_6X6_250) → posición exacta
     del gripper en frame cámara
  3. Detecta cubo azul (HSV) en el mismo frame
  4. Delta gripper→cubo en frame cámara → corrección de arm/lift
  5. Wrist servo (D405) para alineación final
  6. Bajar, cerrar, levantar, colocar sobre rojo

Usage:
    MUJOCO_GL=glfw  python cube_stack_ssh.py   # con ventana
    MUJOCO_GL=egl   python cube_stack_ssh.py   # headless SSH
"""

import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import sys, time, threading
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
HEADLESS  = True
VIDEO_FPS = 10
VIDEO_W, VIDEO_H = 640, 480

CAMERAS = [
    StretchCameras.cam_d435i_rgb,
    StretchCameras.cam_d435i_depth,
    StretchCameras.cam_d405_rgb,
    StretchCameras.cam_overhead,
]

# ── Intrínsecos D435i (raw, sin rotar: 424×240) ───────────────────────────────
D435I_FX, D435I_FY = 303.07, 303.06
D435I_CX, D435I_CY = 122.79, 210.94
D435I_W,  D435I_H  = 424, 240
D435I_DEPTH_SCALE  = 1e-3
D435I_K = np.array([[D435I_FX, 0, D435I_CX],
                     [0, D435I_FY, D435I_CY],
                     [0, 0,        1       ]], dtype=np.float32)
D435I_DIST = np.zeros(5, dtype=np.float32)

# ── ArUco del brazo (ya en el modelo) ────────────────────────────────────────
# arm_top_wrist_aruco_sticker.png → DICT_6X6_250 ID=133
# right_finger_aruco.png          → DICT_6X6_250 ID=201
# left_finger_aruco.png           → DICT_6X6_250 ID=200
ARUCO_DICT_ID    = cv2.aruco.DICT_6X6_250
ARUCO_WRIST_ID   = 133    # muñeca superior (más visible desde arriba)
ARUCO_MARKER_M   = 0.04   # tamaño físico del sticker ≈ 4 cm
_aruco_dict      = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
_aruco_params    = cv2.aruco.DetectorParameters()
_aruco_detector  = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

# Posiciones fallback de los cubos (del XML, tras caer en la mesa)
CUBE_BLUE_FB = np.array([-0.04, -0.55, 0.52])
CUBE_RED_FB  = np.array([ 0.12, -0.55, 0.52])

GRIPPER_OPEN  =  0.04
GRIPPER_CLOSE = -0.015
ARM_PROBE_M   = 0.12

HEAD_PAN_LIM  = (-4.04,  1.73)
HEAD_TILT_LIM = (-1.53,  0.00)

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Actuadores ────────────────────────────────────────────────────────────────
def move(sim, act, pos, timeout=8.0):
    sim.move_to(act, pos)
    sim.wait_while_is_moving(act, timeout=timeout)

def ee_pos(sim): return sim.get_ee_pose()[:3, 3].copy()

# ── Cabeza: apuntar a un punto ────────────────────────────────────────────────
def point_head_at(sim, target_xyz):
    """Calcula y comanda pan+tilt para mirar directamente a target_xyz."""
    st    = sim.pull_status()
    theta = st.base.theta
    head  = np.array([st.base.x - 0.1*np.cos(theta),
                      st.base.y - 0.1*np.sin(theta),
                      st.lift.pos + 0.20])
    v    = target_xyz - head
    fwd  = np.array([ np.cos(theta),  np.sin(theta)])
    lft  = np.array([-np.sin(theta),  np.cos(theta)])
    vf   = v[0]*fwd[0] + v[1]*fwd[1]
    vl   = v[0]*lft[0] + v[1]*lft[1]
    pan  = float(np.clip(np.arctan2(vl, vf),        *HEAD_PAN_LIM))
    tilt = float(np.clip(np.arctan2(v[2], np.hypot(vf, vl)), *HEAD_TILT_LIM))
    log(f"  → cabeza pan={pan:.2f} tilt={tilt:.2f}")
    sim.move_to(Actuators.head_pan,  pan)
    sim.move_to(Actuators.head_tilt, tilt)
    sim.wait_while_is_moving(Actuators.head_pan,  timeout=4)
    sim.wait_while_is_moving(Actuators.head_tilt, timeout=4)
    time.sleep(0.25)

# ── Cámara: leer frames raw ───────────────────────────────────────────────────
def get_head_frames(sim):
    """Devuelve (bgr_raw, depth_raw) de la cámara de cabeza sin rotar."""
    cam = sim.pull_camera_data()
    rgb = cam.get_camera_data(StretchCameras.cam_d435i_rgb,
                              auto_rotate=False, auto_correct_rgb=False)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        dep = cam.get_camera_data(StretchCameras.cam_d435i_depth,
                                  auto_rotate=False, auto_correct_rgb=False)
    except Exception:
        dep = None
    return bgr, dep

# ── Backprojection pixel→3D en frame cámara ───────────────────────────────────
def backproject(u, v, depth_m):
    x = (u - D435I_CX) / D435I_FX * depth_m
    y = (v - D435I_CY) / D435I_FY * depth_m
    return np.array([x, y, depth_m])

def depth_sample(depth_raw, u, v, r=4):
    if depth_raw is None: return 0.0
    h, w = depth_raw.shape[:2]
    vals = [depth_raw[py, px]
            for dv in range(-r, r+1) for du in range(-r, r+1)
            if 0 <= (py:=v+dv) < h and 0 <= (px:=u+du) < w and depth_raw[py,px] > 0]
    return float(np.median(vals)) if vals else 0.0

# ── Pose de la cámara en frame mundo (para transformar deltas) ────────────────
def get_head_cam_R(sim):
    """Matriz de rotación cámara→mundo (aproximada desde articulaciones)."""
    st    = sim.pull_status()
    theta = st.base.theta
    pan   = st.head_pan.pos
    tilt  = st.head_tilt.pos
    heading = theta + pan
    # Ejes de la cámara en mundo
    z_cam = np.array([np.cos(heading)*np.cos(tilt),
                      np.sin(heading)*np.cos(tilt),
                     -np.sin(tilt)])
    x_cam = np.array([-np.sin(heading), np.cos(heading), 0.0])
    y_cam = np.cross(z_cam, x_cam)
    return np.column_stack([x_cam, y_cam, z_cam])   # R_world_from_cam

# ── Detección de color (HSV) ──────────────────────────────────────────────────
BLUE_LO, BLUE_HI = np.array([100,150,80]),  np.array([130,255,255])
RED_LO1, RED_HI1 = np.array([0,  140,80]),  np.array([ 10,255,255])
RED_LO2, RED_HI2 = np.array([170,140,80]),  np.array([180,255,255])

def find_cube_pixel(bgr, color, skip_top=0.30):
    if bgr is None or bgr.ndim != 3: return None
    h, w = bgr.shape[:2]
    y0   = int(h * skip_top)
    hsv  = cv2.cvtColor(bgr[y0:], cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, BLUE_LO, BLUE_HI) if color == "blue"
            else cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                                cv2.inRange(hsv, RED_LO2, RED_HI2)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 100: return None
    M = cv2.moments(best)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]) + y0)

# ── Detección ArUco del brazo ─────────────────────────────────────────────────
def detect_wrist_aruco(bgr):
    """
    Detecta el ArUco de muñeca (ID=133, DICT_6X6_250).
    Devuelve (pixel_centro, corners_4x2) o (None, None).
    """
    corners, ids, _ = _aruco_detector.detectMarkers(bgr)
    if ids is None: return None, None
    flat = ids.flatten()
    if ARUCO_WRIST_ID not in flat: return None, None
    idx  = np.where(flat == ARUCO_WRIST_ID)[0][0]
    pts  = corners[idx].reshape(4, 2)
    cx   = int(pts[:, 0].mean())
    cy   = int(pts[:, 1].mean())
    return (cx, cy), pts

# ── Alineación principal: ArUco muñeca + detección cubo ──────────────────────
def aruco_align_to_cube(sim, color, arm_dir, max_iters=8, tol_px=20):
    """
    Alineación óptica usando ArUco de muñeca como referencia del gripper.

    Con wrist_pitch≈-0.3 (muñeca casi horizontal) el ArUco superior
    es visible desde la cámara de cabeza mirando hacia abajo.

    En cada iteración:
      - Detecta ArUco muñeca → posición gripper en frame cámara
      - Detecta cubo azul   → posición cubo en frame cámara
      - Calcula delta en frame cámara → convierte a mundo → ajusta arm/lift
    """
    log("  Alineación ArUco+visión...")

    # Wrist en ángulo intermedio para que el ArUco superior sea visible
    move(sim, Actuators.wrist_pitch, -0.3, timeout=4)
    time.sleep(0.3)

    for it in range(max_iters):
        # Apuntar cabeza al EE actual
        point_head_at(sim, ee_pos(sim))
        time.sleep(0.2)

        bgr, dep = get_head_frames(sim)

        # ── Detectar ArUco de muñeca ──────────────────────────────────────
        aruco_px, _ = detect_wrist_aruco(bgr)
        if aruco_px is None:
            log(f"  iter {it}: ArUco muñeca NO visto — ajustando tilt")
            st = sim.pull_status()
            sim.move_to(Actuators.head_tilt,
                        float(np.clip(st.head_tilt.pos - 0.05, *HEAD_TILT_LIM)))
            time.sleep(0.3)
            continue

        # ── Detectar cubo ─────────────────────────────────────────────────
        cube_px = find_cube_pixel(bgr, color, skip_top=0.0)
        if cube_px is None:
            log(f"  iter {it}: cubo NO visto")
            break

        # ── Delta en pixels ───────────────────────────────────────────────
        du = cube_px[0] - aruco_px[0]
        dv = cube_px[1] - aruco_px[1]
        log(f"  iter {it}: ArUco={aruco_px} cubo={cube_px} Δpx=({du},{dv})")

        if abs(du) < tol_px and abs(dv) < tol_px:
            log(f"  Alineado (error < {tol_px}px)")
            break

        # ── Profundidades ─────────────────────────────────────────────────
        d_aruco_raw = depth_sample(dep, aruco_px[0], aruco_px[1])
        d_cube_raw  = depth_sample(dep, cube_px[0],  cube_px[1])
        d_aruco = d_aruco_raw * D435I_DEPTH_SCALE if d_aruco_raw > 0 else 0.5
        d_cube  = d_cube_raw  * D435I_DEPTH_SCALE if d_cube_raw  > 0 else d_aruco

        # ── Posiciones 3D en frame cámara ─────────────────────────────────
        p_aruco = backproject(aruco_px[0], aruco_px[1], d_aruco)
        p_cube  = backproject(cube_px[0],  cube_px[1],  d_cube)
        delta_cam = p_cube - p_aruco
        log(f"  delta_cam={np.round(delta_cam, 3)}")

        # ── Convertir a frame mundo y aplicar corrección ──────────────────
        R          = get_head_cam_R(sim)
        delta_w    = R @ delta_cam
        log(f"  delta_world={np.round(delta_w, 3)}")

        # arm: componente a lo largo de la dirección del brazo
        d_arm  = float(np.dot(delta_w[:2], arm_dir)) * 0.7   # ganancia < 1
        # lift: componente vertical
        d_lift = float(delta_w[2]) * 0.7

        st = sim.pull_status()
        new_arm  = float(np.clip(st.arm.pos  + d_arm,  0.0, 0.52))
        new_lift = float(np.clip(st.lift.pos + d_lift, 0.05, 1.0))
        log(f"  → arm {st.arm.pos:.3f}→{new_arm:.3f}  lift {st.lift.pos:.3f}→{new_lift:.3f}")
        move(sim, Actuators.arm,  new_arm,  timeout=5)
        move(sim, Actuators.lift, new_lift, timeout=5)
        time.sleep(0.3)

    # Restaurar wrist para el agarre
    move(sim, Actuators.wrist_pitch, -0.9, timeout=4)

# ── Servo fino con cámara de muñeca (D405) ───────────────────────────────────
def servo_wrist_to_cube(sim, color, max_iters=25, tol=0.10):
    """P-controller con D405: centra cubo ajustando arm (err_u) y lift (err_v)."""
    KP = 0.02
    log("  Servo muñeca D405...")
    for _ in range(max_iters):
        try:
            frame = sim.pull_camera_data().get_camera_data(StretchCameras.cam_d405_rgb)
        except Exception:
            time.sleep(0.1); continue
        h, w  = frame.shape[:2]
        px    = find_cube_pixel(frame, color, skip_top=0.0)
        if px is None: return False
        eu, ev = px[0]/w - 0.5, px[1]/h - 0.5
        if abs(eu) < tol and abs(ev) < tol:
            log(f"  D405 centrado {px}")
            return True
        st = sim.pull_status()
        sim.move_to(Actuators.arm,  float(np.clip(st.arm.pos  + KP*eu, 0.0, 0.52)))
        sim.move_to(Actuators.lift, float(np.clip(st.lift.pos - KP*ev, 0.05, 1.0)))
        time.sleep(0.15)
    return False

# ── Geometría del brazo (calibración empírica) ────────────────────────────────
def rotate_2d(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1]])

def probe_arm_geometry(sim):
    log("Calibrando geometría del brazo...")
    move(sim, Actuators.lift, 0.5, timeout=10)
    move(sim, Actuators.arm,  0.0, timeout=10)
    time.sleep(0.5)
    ee0 = ee_pos(sim)
    th0 = sim.pull_status().base.theta
    log(f"  EE home: {np.round(ee0,3)}  theta={th0:.3f}")
    # lift ratio
    move(sim, Actuators.lift, 0.6, timeout=6)
    dz = (ee_pos(sim)[2] - ee0[2]) / 0.1
    move(sim, Actuators.lift, 0.5, timeout=6)
    # arm direction
    move(sim, Actuators.arm, ARM_PROBE_M, timeout=8)
    time.sleep(0.4)
    d_xy  = ee_pos(sim)[:2] - ee0[:2]
    adir  = d_xy / (np.linalg.norm(d_xy) + 1e-9)
    move(sim, Actuators.arm, 0.0, timeout=8)
    time.sleep(0.3)
    log(f"  dz/lift={dz:.3f}  arm_dir={np.round(adir,3)}")
    return adir, ee_pos(sim), dz, th0

def current_arm_dir(sim, adir_cal, th_cal):
    return rotate_2d(adir_cal, sim.pull_status().base.theta - th_cal)

# ── Control de base ───────────────────────────────────────────────────────────
def move_base_to(sim, tx, ty, tol=0.06, timeout=20.0):
    KP_L, KP_A = 1.5, 3.0
    t0 = time.time()
    while time.time()-t0 < timeout:
        st = sim.pull_status()
        dx, dy = tx-st.base.x, ty-st.base.y
        dist = np.hypot(dx, dy)
        if dist < tol: break
        ae = (np.arctan2(dy, dx)-st.base.theta+np.pi) % (2*np.pi) - np.pi
        vl = 0.0 if abs(ae)>0.4 else min(KP_L*dist, 0.25)
        sim.set_base_velocity(vl, KP_A*ae)
        time.sleep(0.04)
    sim.set_base_velocity(0, 0); time.sleep(0.3)
    st = sim.pull_status()
    log(f"  Base: ({st.base.x:.3f}, {st.base.y:.3f})")

# ── Pick and place ────────────────────────────────────────────────────────────
def pick_and_place(sim, adir_cal, th_cal, dz_per_lift):
    DESIRED_ARM = 0.35

    def refresh():
        move(sim, Actuators.lift, 0.5, timeout=6)
        move(sim, Actuators.arm,  0.0, timeout=6)
        time.sleep(0.3)
        ref  = ee_pos(sim)
        adir = current_arm_dir(sim, adir_cal, th_cal)
        log(f"  EE ref: {np.round(ref,3)} arm_dir: {np.round(adir,3)}")
        return ref, adir

    def joints_for(target, ref, adir, dz_off=0.0):
        lift = float(np.clip(0.5+(target[2]+dz_off-ref[2])/dz_per_lift, 0.05, 1.0))
        arm  = float(np.clip(np.dot(target[:2]-ref[:2], adir), 0.0, 0.50))
        return lift, arm

    # ── 1. Preparar muñeca y abrir gripper ────────────────────────────────────
    move(sim, Actuators.wrist_yaw,    0.0, timeout=5)
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)

    # ── 2. Brazo retractado, detectar cubos ───────────────────────────────────
    log("Fase 1: detectando cubos con cabeza (brazo adentro)...")
    move(sim, Actuators.arm,  0.0, timeout=8)
    move(sim, Actuators.lift, 0.5, timeout=8)
    time.sleep(0.4)

    # Detectar cubo azul
    point_head_at(sim, CUBE_BLUE_FB)
    bgr, dep = get_head_frames(sim)
    cube_blue = CUBE_BLUE_FB.copy()
    px_b = find_cube_pixel(bgr, "blue")
    if px_b:
        d_raw = depth_sample(dep, px_b[0], px_b[1])
        if d_raw > 0:
            depth_m = d_raw * D435I_DEPTH_SCALE
            TABLE_Z = 0.52
            R = get_head_cam_R(sim)
            # Posición de la cabeza
            st = sim.pull_status(); theta = st.base.theta
            head = np.array([st.base.x-0.1*np.cos(theta),
                             st.base.y-0.1*np.sin(theta),
                             st.lift.pos+0.20])
            # ray-plane con depth
            p_cam   = backproject(px_b[0], px_b[1], depth_m)
            p_world = head + R @ p_cam
            if -1.0 < p_world[0] < 1.0 and -1.5 < p_world[1] < 0.0 and 0.3 < p_world[2] < 0.8:
                cube_blue = p_world
    log(f"  Cubo azul: {np.round(cube_blue,3)}")

    # Detectar cubo rojo
    point_head_at(sim, CUBE_RED_FB)
    bgr, dep = get_head_frames(sim)
    cube_red = CUBE_RED_FB.copy()
    px_r = find_cube_pixel(bgr, "red")
    if px_r:
        d_raw = depth_sample(dep, px_r[0], px_r[1])
        if d_raw > 0:
            depth_m = d_raw * D435I_DEPTH_SCALE
            R = get_head_cam_R(sim)
            st = sim.pull_status(); theta = st.base.theta
            head = np.array([st.base.x-0.1*np.cos(theta),
                             st.base.y-0.1*np.sin(theta),
                             st.lift.pos+0.20])
            p_cam   = backproject(px_r[0], px_r[1], depth_m)
            p_world = head + R @ p_cam
            if -1.0 < p_world[0] < 1.0 and -1.5 < p_world[1] < 0.0 and 0.3 < p_world[2] < 0.8:
                cube_red = p_world
    log(f"  Cubo rojo: {np.round(cube_red,3)}")

    # ── 3. Posicionar base ────────────────────────────────────────────────────
    log("Fase 2: posicionando base frente al cubo azul...")
    point_head_at(sim, cube_blue)
    adir = current_arm_dir(sim, adir_cal, th_cal)
    move_base_to(sim, cube_blue[0]-adir[0]*DESIRED_ARM,
                      cube_blue[1]-adir[1]*DESIRED_ARM)

    # ── 4. Hover inicial (estimado) ───────────────────────────────────────────
    log("Fase 3: hover sobre cubo azul (posición estimada)...")
    ee_ref, adir = refresh()
    lift_h, arm_h = joints_for(cube_blue, ee_ref, adir, dz_off=0.08)
    log(f"  lift={lift_h:.3f}  arm={arm_h:.3f}")
    move(sim, Actuators.lift, lift_h, timeout=8)
    move(sim, Actuators.arm,  arm_h,  timeout=8)
    log(f"  EE real: {np.round(ee_pos(sim),3)}")

    # ── 5. ALINEACIÓN ARUCO + VISIÓN ──────────────────────────────────────────
    # Cabeza mira al brazo, detecta ArUco de muñeca Y cubo en el mismo frame,
    # itera hasta que gripper esté sobre el cubo.
    log("Fase 4: alineación ArUco muñeca ↔ cubo azul...")
    aruco_align_to_cube(sim, "blue", adir)

    # ── 6. Servo fino D405 ────────────────────────────────────────────────────
    log("Fase 5: servo fino con D405...")
    move(sim, Actuators.wrist_pitch, -0.9, timeout=4)
    point_head_at(sim, ee_pos(sim))
    servo_wrist_to_cube(sim, "blue")

    # ── 7. Bajar y agarrar ────────────────────────────────────────────────────
    log("Fase 6: bajando y agarrando...")
    st = sim.pull_status()
    move(sim, Actuators.lift, st.lift.pos - 0.07, timeout=6)
    time.sleep(0.4)
    move(sim, Actuators.gripper, GRIPPER_CLOSE, timeout=4)
    time.sleep(0.6)

    # ── 8. Levantar ───────────────────────────────────────────────────────────
    st = sim.pull_status()
    lift_carry = float(np.clip(st.lift.pos + 0.22, 0.05, 1.0))
    log("Fase 7: levantando cubo azul...")
    move(sim, Actuators.lift, lift_carry, timeout=8)
    time.sleep(0.5)

    # ── 9. Ir al cubo rojo ────────────────────────────────────────────────────
    log("Fase 8: moviéndose al cubo rojo...")
    adir = current_arm_dir(sim, adir_cal, th_cal)
    move_base_to(sim, cube_red[0]-adir[0]*DESIRED_ARM,
                      cube_red[1]-adir[1]*DESIRED_ARM)

    # Hover sobre rojo
    move(sim, Actuators.arm, 0.0, timeout=6)
    ee_ref, adir = refresh()
    lift_o, arm_r = joints_for(cube_red, ee_ref, adir, dz_off=0.10)
    move(sim, Actuators.lift, lift_carry, timeout=6)
    move(sim, Actuators.arm,  arm_r,      timeout=8)
    move(sim, Actuators.lift, lift_o,     timeout=8)

    # Servo fino D405 para el rojo
    servo_wrist_to_cube(sim, "red")

    # ── 10. Soltar ────────────────────────────────────────────────────────────
    log("Fase 9: colocando cubo azul sobre rojo...")
    st = sim.pull_status()
    move(sim, Actuators.lift, st.lift.pos - 0.05, timeout=6)
    time.sleep(0.4)
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)
    time.sleep(0.5)

    # ── 11. Retirar ───────────────────────────────────────────────────────────
    log("Fase 10: retirando...")
    move(sim, Actuators.lift, lift_carry, timeout=6)
    move(sim, Actuators.arm,  0.0,        timeout=6)
    time.sleep(1.0)
    log("¡Completado!")

# ── Video compuesto 2×2 ───────────────────────────────────────────────────────
CELL_W, CELL_H = VIDEO_W//2, VIDEO_H//2

def _colorize_depth(dep):
    if dep is None: return np.zeros((D435I_H, D435I_W, 3), dtype=np.uint8)
    n = cv2.normalize(dep, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(n, cv2.COLORMAP_TURBO)

class VideoRecorder:
    def __init__(self, sim, path):
        self.sim = sim
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (VIDEO_W, VIDEO_H))
        self.label  = ""
        self._running = False

    def _build(self, cam):
        # Head RGB con detección
        try:
            rgb_r = cam.get_camera_data(StretchCameras.cam_d435i_rgb,
                                        auto_rotate=False, auto_correct_rgb=False)
            head  = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
        except Exception:
            head = np.zeros((D435I_H, D435I_W, 3), dtype=np.uint8)

        # Dibujar ArUco
        aruco_px, _ = detect_wrist_aruco(head)
        if aruco_px:
            cv2.circle(head, aruco_px, 16, (0, 255, 255), 3)
            cv2.putText(head, "ArUco", (aruco_px[0]+8, aruco_px[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        for col, bgr_c in [("blue",(255,80,0)), ("red",(0,50,255))]:
            px = find_cube_pixel(head, col)
            if px: cv2.circle(head, px, 12, bgr_c, 3)
        cv2.putText(head, "Head RGB + ArUco", (4,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # Head Depth
        try:
            dep_r = cam.get_camera_data(StretchCameras.cam_d435i_depth,
                                        auto_rotate=False, auto_correct_rgb=False)
            depth = _colorize_depth(dep_r)
        except Exception:
            depth = np.zeros((D435I_H, D435I_W, 3), dtype=np.uint8)
        cv2.putText(depth, "Head Depth", (4,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # Wrist D405
        try:
            wrist = cam.get_camera_data(StretchCameras.cam_d405_rgb).copy()
        except Exception:
            wrist = np.zeros((270, 480, 3), dtype=np.uint8)
        px = find_cube_pixel(wrist, "blue", skip_top=0.0)
        if px: cv2.circle(wrist, px, 12, (0,255,0), 3)
        cv2.putText(wrist, "Wrist D405", (4,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # Overhead
        try:
            over = cam.get_camera_data(StretchCameras.cam_overhead).copy()
        except Exception:
            over = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        if self.label:
            cv2.putText(over, self.label, (10, VIDEO_H//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2)
        cv2.putText(over, "Overhead", (4,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        grid = np.vstack([
            np.hstack([cv2.resize(head,  (CELL_W,CELL_H)),
                       cv2.resize(depth, (CELL_W,CELL_H))]),
            np.hstack([cv2.resize(wrist, (CELL_W,CELL_H)),
                       cv2.resize(over,  (CELL_W,CELL_H))]),
        ])
        return grid

    def _loop(self):
        interval = 1.0/VIDEO_FPS
        while self._running and self.sim.is_running():
            t0 = time.perf_counter()
            try:
                self.writer.write(self._build(self.sim.pull_camera_data()))
            except Exception:
                pass
            time.sleep(max(0, interval-(time.perf_counter()-t0)))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        time.sleep(0.3)
        self.writer.release()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Cube Stack — ArUco arm alignment + vision detection")
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL','no seteado')}")
    print(f"  Escena    = {SCENE_XML}")
    print(f"  Video     = {VIDEO_OUT}")
    print("="*60)

    sim = StretchMujocoSimulator(
        scene_xml_path=SCENE_XML, cameras_to_use=CAMERAS, camera_hz=10)

    log("Iniciando simulación...")
    sim.start(headless=HEADLESS)
    if not sim.is_running():
        log("ERROR: simulación no arrancó"); sys.exit(1)

    rec = VideoRecorder(sim, VIDEO_OUT)
    rec.start()

    try:
        rec.label = "Estabilizando..."
        log("Esperando que los cubos caigan...")
        time.sleep(3.0)

        rec.label = "Calibrando..."
        adir, _, dz, th0 = probe_arm_geometry(sim)

        rec.label = "Pick & place (ArUco+visión)..."
        pick_and_place(sim, adir, th0, dz)

        rec.label = "Completado"
        time.sleep(3.0)

    except KeyboardInterrupt:
        log("Interrumpido.")
    finally:
        rec.stop()
        log(f"Video: {VIDEO_OUT}")
        sim.stop()

if __name__ == "__main__":
    main()
