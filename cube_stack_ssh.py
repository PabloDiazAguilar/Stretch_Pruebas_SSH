"""
Cube stacking demo — vision-based using all cameras.

Head camera (D435i RGB+Depth): localiza cubos en 3D
Wrist camera (D405 RGB):       confirma posición antes de agarrar
Overhead camera:                graba video en tercera persona

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

# Intrínsecos de la cámara de cabeza D435i (imagen sin rotar, raw)
# Extraídos de sim.py / initial_camera_settings
D435I_FX   = 303.07
D435I_FY   = 303.06
D435I_CX   = 122.79
D435I_CY   = 210.94
D435I_W    = 424
D435I_H    = 240
D435I_DEPTH_SCALE = 1e-3    # metros por unidad de profundidad

# Fallback si la detección visual falla (posiciones del XML)
CUBE_BLUE_FALLBACK = np.array([-0.04, -0.55, 0.52])
CUBE_RED_FALLBACK  = np.array([ 0.12, -0.55, 0.52])

# Gripper
GRIPPER_OPEN  =  0.04
GRIPPER_CLOSE = -0.015
ARM_PROBE_DIST = 0.12

# Nombres candidatos para get_link_pose de la cámara de cabeza
HEAD_CAM_LINKS = [
    "camera_color_optical_frame",
    "camera_color_frame",
    "camera_link",
    "link_head_tilt",
]

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Actuadores ────────────────────────────────────────────────────────────────

def move(sim, actuator, pos, timeout=8.0):
    sim.move_to(actuator, pos)
    sim.wait_while_is_moving(actuator, timeout=timeout)

def ee_pos(sim):
    return sim.get_ee_pose()[:3, 3].copy()

# ── Detección de color (HSV) ──────────────────────────────────────────────────

# S mínimo alto (150) para distinguir azul brillante del cubo vs celeste del cielo
BLUE_LO, BLUE_HI = np.array([100, 150, 80]),  np.array([130, 255, 255])
RED_LO1, RED_HI1 = np.array([  0, 140, 80]),  np.array([ 10, 255, 255])
RED_LO2, RED_HI2 = np.array([170, 140, 80]),  np.array([180, 255, 255])

def find_cube_pixel(frame_bgr, color, skip_top=0.35):
    """
    Detecta el cubo por color HSV y devuelve pixel central (u,v) en coordenadas
    del frame completo, o None.
    skip_top: fracción superior del frame a ignorar (evita cielo/fondo).
    """
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    h, w = frame_bgr.shape[:2]
    roi_y = int(h * skip_top)
    roi = frame_bgr[roi_y:, :]                    # ignorar parte superior
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color == "blue":
        mask = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    else:
        mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                              cv2.inRange(hsv, RED_LO2, RED_HI2))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 100:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    # Ajustar coordenada Y de vuelta al frame completo
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) + roi_y)

# ── Servo visual de cabeza ────────────────────────────────────────────────────

# Límites de las articulaciones de la cabeza
HEAD_PAN_LIMITS  = (-4.04, 1.73)
HEAD_TILT_LIMITS = (-1.53, 0.00)   # 0.00 = horizontal, -1.53 = abajo


def _head_sees_cube(sim, color):
    """Devuelve pixel si la cabeza ve el cubo, None si no."""
    try:
        frame = sim.pull_camera_data().get_camera_data(StretchCameras.cam_d435i_rgb)
        return find_cube_pixel(frame, color)
    except Exception:
        return None


def sweep_head_for_cube(sim, color):
    """
    Barrido sistemático pan+tilt para encontrar el cubo.
    Patrón: para cada nivel de tilt (de menos a más abajo), barre pan izq→der.
    Devuelve True si lo encontró (cabeza ya apuntando hacia él).
    """
    tilt_levels = [-0.5, -0.7, -0.9, -1.1, -1.3]
    pan_sweep   = [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2]

    log(f"  Barrido pan+tilt buscando cubo {color}...")
    for tilt in tilt_levels:
        sim.move_to(Actuators.head_tilt, tilt)
        sim.wait_while_is_moving(Actuators.head_tilt, timeout=3)
        for pan in pan_sweep:
            sim.move_to(Actuators.head_pan, pan)
            sim.wait_while_is_moving(Actuators.head_pan, timeout=2)
            time.sleep(0.15)
            px = _head_sees_cube(sim, color)
            if px is not None:
                log(f"  Cubo {color} encontrado: tilt={tilt:.1f} pan={pan:.1f} px={px}")
                return True
    log(f"  Barrido completo: cubo {color} NO encontrado")
    return False


def servo_head_to_cube(sim, color, tolerance=0.08, max_iters=50):
    """
    1. Si no lo ve → barrido completo pan+tilt.
    2. Una vez visible → P-controller hasta centrar.
    Devuelve (True, pixel) si convergió, (False, None) si no.
    """
    KP = 0.45
    log(f"  Servo cabeza → cubo {color}...")

    # Búsqueda inicial si no está visible
    if _head_sees_cube(sim, color) is None:
        if not sweep_head_for_cube(sim, color):
            return False, None

    for _ in range(max_iters):
        try:
            frame = sim.pull_camera_data().get_camera_data(StretchCameras.cam_d435i_rgb)
        except Exception:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        pixel = find_cube_pixel(frame, color)

        if pixel is None:
            # Perdió de vista: re-buscar
            if not sweep_head_for_cube(sim, color):
                return False, None
            continue

        err_u = pixel[0] / w - 0.5
        err_v = pixel[1] / h - 0.5

        if abs(err_u) < tolerance and abs(err_v) < tolerance:
            log(f"  Servo cabeza: {color} centrado {pixel} err=({err_u:.2f},{err_v:.2f})")
            return True, pixel

        st = sim.pull_status()
        sim.move_to(Actuators.head_pan,
                    float(np.clip(st.head_pan.pos  - KP * err_u, *HEAD_PAN_LIMITS)))
        sim.move_to(Actuators.head_tilt,
                    float(np.clip(st.head_tilt.pos - KP * err_v, *HEAD_TILT_LIMITS)))
        time.sleep(0.12)

    log(f"  Servo cabeza: no convergió para {color}")
    return False, None


def start_head_tracking(sim, color, stop_event):
    """
    Hilo de fondo: mantiene la cabeza apuntando al cubo mientras el brazo se mueve.
    stop_event.set() para detenerlo.
    """
    KP = 0.3

    def _track():
        while not stop_event.is_set():
            try:
                cam_data = sim.pull_camera_data()
                head_rgb = cam_data.get_camera_data(StretchCameras.cam_d435i_rgb)
                h, w = head_rgb.shape[:2]
                pixel = find_cube_pixel(head_rgb, color)
                if pixel:
                    err_u = pixel[0] / w - 0.5
                    err_v = pixel[1] / h - 0.5
                    st = sim.pull_status()
                    sim.move_to(Actuators.head_pan,
                                float(np.clip(st.head_pan.pos  - KP * err_u,
                                              *HEAD_PAN_LIMITS)))
                    sim.move_to(Actuators.head_tilt,
                                float(np.clip(st.head_tilt.pos - KP * err_v,
                                              *HEAD_TILT_LIMITS)))
            except Exception:
                pass
            time.sleep(0.12)

    t = threading.Thread(target=_track, daemon=True)
    t.start()
    return t


# ── Localización 3D con cámara de cabeza ──────────────────────────────────────

def get_head_cam_pose(sim):
    """Obtiene la pose de la cámara de cabeza en frame mundo."""
    for name in HEAD_CAM_LINKS:
        try:
            pose = sim.get_link_pose(name)
            log(f"  Pose de cámara obtenida con link: '{name}'")
            return pose
        except Exception:
            continue
    log("  WARN: no se pudo obtener pose de cámara, usando posición del EE como proxy")
    # Proxy: la cámara está aproximadamente donde está el EE pero en la cabeza
    # Construir pose aproximada desde el estado del robot
    st = sim.pull_status()
    theta = st.base.theta
    bx, by = st.base.x, st.base.y
    # Altura aproximada de la cámara (lift + offset fijo ≈ 0.15m por encima del lift)
    cam_z = st.lift.pos + 0.15
    # Rotación de la cabeza: base_theta + head_pan, y head_tilt
    pan  = st.head_pan.pos
    tilt = st.head_tilt.pos
    heading = theta + pan
    # Pose simple (sin FK completo, aproximada)
    pose = np.eye(4)
    pose[0, 3] = bx
    pose[1, 3] = by
    pose[2, 3] = cam_z
    # Dirección de mirada: heading en xy, tilt en z
    cx_ = np.cos(heading) * np.cos(tilt)
    cy_ = np.sin(heading) * np.cos(tilt)
    cz_ = -np.sin(tilt)
    z_ax = np.array([cx_, cy_, cz_])
    x_ax = np.array([-np.sin(heading), np.cos(heading), 0])
    y_ax = np.cross(z_ax, x_ax)
    pose[:3, 0] = x_ax
    pose[:3, 1] = y_ax
    pose[:3, 2] = z_ax
    return pose


def depth_sample(depth_img, u, v, radius=4):
    """Muestra mediana de profundidad alrededor del pixel (u,v)."""
    h, w = depth_img.shape[:2]
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            py, px = v + dv, u + du
            if 0 <= py < h and 0 <= px < w:
                d = depth_img[py, px]
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def detect_cube_3d(sim, color, fallback_xyz):
    """
    Usa la cámara de cabeza (D435i RGB + depth) para detectar un cubo por color
    y calcular su posición 3D en frame mundo.

    Pasos:
      1. Lee imagen RGB raw (sin rotar) y depth raw
      2. Detecta el cubo por HSV en la imagen BGR
      3. Muestrea profundidad en el pixel detectado
      4. Convierte pixel+depth a coordenadas de cámara (backprojection)
      5. Transforma a frame mundo con la pose de la cámara
    """
    log(f"  Detectando cubo {color} con cámara de cabeza...")

    # 1. Servo: centrar cubo en la imagen antes de medir
    servo_ok, _ = servo_head_to_cube(sim, color)
    if not servo_ok:
        log(f"  Servo falló — usando posición de fallback")
        return fallback_xyz.copy()

    # 2. Leer imágenes raw (sin rotar) para backprojection con intrínsecos correctos
    try:
        cam_data = sim.pull_camera_data()
        rgb_raw = cam_data.get_camera_data(
            StretchCameras.cam_d435i_rgb,
            auto_rotate=False, auto_correct_rgb=False)
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        depth_raw = cam_data.get_camera_data(
            StretchCameras.cam_d435i_depth,
            auto_rotate=False, auto_correct_rgb=False)
    except Exception as e:
        log(f"  Error leyendo cámara: {e}")
        return fallback_xyz.copy()

    # Detectar pixel en imagen raw (mismo cubo que centró el servo)
    pixel = find_cube_pixel(bgr_raw, color)
    if pixel is None:
        log(f"  Cubo {color} NO detectado en raw — usando fallback")
        return fallback_xyz.copy()

    u, v = pixel
    log(f"  Cubo {color} en pixel ({u}, {v}) — imagen {D435I_W}x{D435I_H}")

    # Profundidad en el pixel
    d_raw = depth_sample(depth_raw, u, v)
    if d_raw <= 0:
        log(f"  Sin profundidad válida en ({u},{v}) — usando fallback")
        return fallback_xyz.copy()

    depth_m = d_raw * D435I_DEPTH_SCALE
    log(f"  Profundidad al cubo {color}: {depth_m:.3f} m")

    # Backprojection: pixel → coordenadas de cámara (frame óptico)
    x_cam = (u - D435I_CX) / D435I_FX * depth_m
    y_cam = (v - D435I_CY) / D435I_FY * depth_m
    z_cam = depth_m
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    log(f"  Punto en frame cámara: [{x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}]")

    # Transformar a frame mundo
    cam_pose = get_head_cam_pose(sim)
    p_world = (cam_pose @ p_cam)[:3]
    log(f"  Cubo {color} en frame mundo: {np.round(p_world, 3)}")

    # Sanidad: z debe estar entre 0.3 y 0.8 (altura razonable de cubo en mesa)
    if not (0.3 < p_world[2] < 0.8):
        log(f"  z={p_world[2]:.3f} fuera de rango — usando fallback")
        return fallback_xyz.copy()

    return p_world

# ── Verificación con cámara de muñeca ────────────────────────────────────────

def check_wrist_cam(sim, color):
    """Lee D405 y loguea si ve el cubo. Devuelve pixel o None."""
    try:
        wrist_rgb = sim.pull_camera_data().get_camera_data(StretchCameras.cam_d405_rgb)
        pixel = find_cube_pixel(wrist_rgb, color, skip_top=0.0)
        if pixel:
            h, w = wrist_rgb.shape[:2]
            log(f"  Muñeca D405: cubo {color} en {pixel} "
                f"err=({pixel[0]/w-0.5:.2f}, {pixel[1]/h-0.5:.2f})")
        else:
            log(f"  Muñeca D405: cubo {color} NO visible")
        return pixel
    except Exception as e:
        log(f"  Error cámara muñeca: {e}")
        return None


def servo_wrist_to_cube(sim, color, max_iters=35, tolerance=0.10):
    """
    Servo fino usando la cámara de muñeca D405.
    Ajusta lift (err vertical) y arm (err horizontal) para centrar el cubo.
    Con wrist_pitch ≈ -0.9 la cámara mira hacia abajo:
      - err_u > 0 (cubo a la derecha en imagen) → extender más arm
      - err_v > 0 (cubo abajo en imagen)        → bajar lift
    Devuelve True si centrado, False si no visible.
    """
    KP_ARM  = 0.02    # m de arm por unidad de error normalizado
    KP_LIFT = 0.02    # m de lift por unidad de error normalizado
    log(f"  Servo muñeca D405 → cubo {color}...")

    for _ in range(max_iters):
        try:
            wrist_rgb = sim.pull_camera_data().get_camera_data(StretchCameras.cam_d405_rgb)
        except Exception:
            time.sleep(0.1)
            continue

        h, w = wrist_rgb.shape[:2]
        pixel = find_cube_pixel(wrist_rgb, color, skip_top=0.0)

        if pixel is None:
            log(f"  Servo muñeca: cubo {color} NO visible")
            return False

        err_u = pixel[0] / w - 0.5
        err_v = pixel[1] / h - 0.5

        if abs(err_u) < tolerance and abs(err_v) < tolerance:
            log(f"  Servo muñeca: {color} centrado {pixel}")
            return True

        st = sim.pull_status()
        new_arm  = float(np.clip(st.arm.pos  + KP_ARM  * err_u, 0.0,  0.52))
        new_lift = float(np.clip(st.lift.pos - KP_LIFT * err_v, 0.05, 1.0))
        sim.move_to(Actuators.arm,  new_arm)
        sim.move_to(Actuators.lift, new_lift)
        time.sleep(0.15)

    log(f"  Servo muñeca: no convergió para {color}")
    return False

# ── Geometría del brazo ───────────────────────────────────────────────────────

def rotate_2d(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def probe_arm_geometry(sim):
    """Mide empíricamente la dirección del brazo y ratio lift→z."""
    log("Calibrando geometría del brazo...")
    move(sim, Actuators.lift, 0.5, timeout=10)
    move(sim, Actuators.arm,  0.0, timeout=10)
    time.sleep(0.5)

    ee0 = ee_pos(sim)
    theta_cal = sim.pull_status().base.theta
    log(f"  EE en lift=0.5 arm=0: {np.round(ee0, 3)}, base_theta={theta_cal:.3f} rad")

    # Probar lift
    move(sim, Actuators.lift, 0.6, timeout=6)
    dz_per_lift = (ee_pos(sim)[2] - ee0[2]) / 0.1
    move(sim, Actuators.lift, 0.5, timeout=6)
    log(f"  dz/d_lift ≈ {dz_per_lift:.3f}")

    # Probar dirección del brazo
    move(sim, Actuators.arm, ARM_PROBE_DIST, timeout=8)
    time.sleep(0.4)
    d_arm = ee_pos(sim)[:2] - ee0[:2]
    arm_dir = d_arm / (np.linalg.norm(d_arm) + 1e-9)
    move(sim, Actuators.arm, 0.0, timeout=8)
    time.sleep(0.3)
    log(f"  Dirección del brazo (world xy): {np.round(arm_dir, 3)}")

    ee_ref = ee_pos(sim)
    return arm_dir, ee_ref, dz_per_lift, theta_cal


def current_arm_dir(sim, arm_dir_cal, theta_cal):
    """Corrige la dirección del brazo según el ángulo actual de la base."""
    delta = sim.pull_status().base.theta - theta_cal
    return rotate_2d(arm_dir_cal, delta)

# ── Control de base ───────────────────────────────────────────────────────────

def move_base_to(sim, tx, ty, tolerance=0.06, timeout=20.0):
    KP_LIN, KP_ANG = 1.5, 3.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = sim.pull_status()
        dx, dy = tx - st.base.x, ty - st.base.y
        dist = np.hypot(dx, dy)
        if dist < tolerance:
            break
        ang_err = (np.arctan2(dy, dx) - st.base.theta + np.pi) % (2 * np.pi) - np.pi
        v_lin = 0.0 if abs(ang_err) > 0.4 else min(KP_LIN * dist, 0.25)
        sim.set_base_velocity(v_lin, KP_ANG * ang_err)
        time.sleep(0.04)
    sim.set_base_velocity(0, 0)
    time.sleep(0.3)
    st = sim.pull_status()
    log(f"  Base final: ({st.base.x:.3f}, {st.base.y:.3f})")

# ── Pick and place ────────────────────────────────────────────────────────────

def pick_and_place(sim, arm_dir_cal, theta_cal, dz_per_lift):

    DESIRED_ARM = 0.35   # extensión de brazo al posicionar la base

    def refresh_refs():
        """Después de mover la base, recalcula EE ref y dirección del brazo."""
        move(sim, Actuators.lift, 0.5, timeout=6)
        move(sim, Actuators.arm,  0.0, timeout=6)
        time.sleep(0.3)
        ref = ee_pos(sim)
        adir = current_arm_dir(sim, arm_dir_cal, theta_cal)
        log(f"  EE ref: {np.round(ref, 3)} | arm_dir: {np.round(adir, 3)}")
        return ref, adir

    def joints_for(target_xyz, ee_ref, adir, dz_offset=0.0):
        """Calcula (lift, arm) para posicionar el EE en target_xyz."""
        dz   = (target_xyz[2] + dz_offset) - ee_ref[2]
        lift = float(np.clip(0.5 + dz / dz_per_lift, 0.05, 1.0))
        arm  = float(np.clip(np.dot(target_xyz[:2] - ee_ref[:2], adir), 0.0, 0.50))
        return lift, arm

    # ── 1. Mirar la mesa con la cámara de cabeza ──────────────────────────────
    log("Fase 1: apuntando cámara de cabeza a la mesa...")
    move(sim, Actuators.head_tilt, -0.85, timeout=5)
    move(sim, Actuators.head_pan,   0.0,  timeout=5)
    time.sleep(0.6)

    # ── 2. Detectar posición 3D del cubo azul con head camera ─────────────────
    log("Fase 2: localizando cubo AZUL con cabeza D435i...")
    cube_blue = detect_cube_3d(sim, "blue", CUBE_BLUE_FALLBACK)

    # ── 3. Detectar posición 3D del cubo rojo con head camera ─────────────────
    log("Fase 3: localizando cubo ROJO con cabeza D435i...")
    cube_red  = detect_cube_3d(sim, "red",  CUBE_RED_FALLBACK)

    log(f"  → Azul: {np.round(cube_blue, 3)}")
    log(f"  → Rojo: {np.round(cube_red,  3)}")

    # ── 4. Preparar muñeca ─────────────────────────────────────────────────────
    move(sim, Actuators.wrist_pitch, -0.9, timeout=5)
    move(sim, Actuators.wrist_yaw,    0.0, timeout=5)
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)

    # ── 5. Posicionar base frente al cubo azul con head tracking activo ───────
    log("Fase 4: posicionando base — head tracking cubo azul...")
    stop_track = threading.Event()
    start_head_tracking(sim, "blue", stop_track)   # cabeza sigue al cubo

    adir_now = current_arm_dir(sim, arm_dir_cal, theta_cal)
    base_tgt = cube_blue[:2] - adir_now * DESIRED_ARM
    move_base_to(sim, base_tgt[0], base_tgt[1])

    stop_track.set()   # parar tracking mientras re-detectamos
    ee_ref, adir_now = refresh_refs()

    # Re-detectar con servo desde nueva posición (más preciso)
    log("  Re-detectando cubo azul con servo desde nueva posición...")
    cube_blue2 = detect_cube_3d(sim, "blue", cube_blue)
    if np.linalg.norm(cube_blue2 - cube_blue) < 0.30:
        cube_blue = cube_blue2
        log(f"  Posición azul refinada: {np.round(cube_blue, 3)}")

    # ── 6. Hover sobre cubo azul con cabeza siguiendo ─────────────────────────
    log("Fase 5: hovering sobre cubo azul...")
    stop_track = threading.Event()
    start_head_tracking(sim, "blue", stop_track)   # tracking mientras brazo sube

    lift_h, arm_h = joints_for(cube_blue, ee_ref, adir_now, dz_offset=0.08)
    log(f"  lift={lift_h:.3f}  arm={arm_h:.3f}")
    move(sim, Actuators.lift, lift_h, timeout=8)
    move(sim, Actuators.arm,  arm_h,  timeout=8)
    time.sleep(0.5)
    log(f"  EE real: {np.round(ee_pos(sim), 3)}")

    stop_track.set()

    # ── 7. Servo muñeca: ajuste fino con D405 ────────────────────────────────
    log("Fase 6: servo fino con cámara de muñeca D405...")
    servo_wrist_to_cube(sim, "blue")

    # ── 8. Bajar al cubo y agarrar ────────────────────────────────────────────
    log("Fase 7: bajando al cubo azul...")
    # Posición post-servo: bajar desde donde quedó el servo
    st = sim.pull_status()
    lift_at_grasp = st.lift.pos - 0.06
    move(sim, Actuators.lift, lift_at_grasp, timeout=6)
    time.sleep(0.5)

    log("Fase 7b: cerrando gripper...")
    move(sim, Actuators.gripper, GRIPPER_CLOSE, timeout=4)
    time.sleep(0.6)

    check_wrist_cam(sim, "blue")

    # ── 9. Levantar cubo ──────────────────────────────────────────────────────
    log("Fase 8: levantando cubo azul...")
    lift_carry = float(np.clip(lift_at_grasp + 0.22, 0.05, 1.0))
    move(sim, Actuators.lift, lift_carry, timeout=8)
    time.sleep(0.5)

    # ── 10. Posicionar base frente al cubo rojo ───────────────────────────────
    log("Fase 9: moviéndose al cubo rojo...")
    adir_now = current_arm_dir(sim, arm_dir_cal, theta_cal)
    base_tgt_r = cube_red[:2] - adir_now * DESIRED_ARM
    move_base_to(sim, base_tgt_r[0], base_tgt_r[1])
    ee_ref, adir_now = refresh_refs()

    # ── 11. Hover sobre cubo rojo ─────────────────────────────────────────────
    log("Fase 10: posicionando sobre cubo rojo...")
    lift_o, arm_r = joints_for(cube_red, ee_ref, adir_now, dz_offset=0.10)
    log(f"  lift={lift_o:.3f}  arm={arm_r:.3f}")
    move(sim, Actuators.lift, lift_carry, timeout=6)   # mantener altura al mover brazo
    move(sim, Actuators.arm,  arm_r, timeout=8)
    move(sim, Actuators.lift, lift_o, timeout=8)
    time.sleep(0.5)
    log(f"  EE real: {np.round(ee_pos(sim), 3)}")

    # ── 12. Soltar ────────────────────────────────────────────────────────────
    log("Fase 11: soltando cubo azul sobre rojo...")
    lift_p, _ = joints_for(cube_red, ee_ref, adir_now, dz_offset=0.06)
    move(sim, Actuators.lift, lift_p, timeout=6)
    time.sleep(0.4)
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4)
    time.sleep(0.5)

    # ── 13. Retirar ───────────────────────────────────────────────────────────
    log("Fase 12: retirando brazo...")
    move(sim, Actuators.lift, lift_carry, timeout=6)
    move(sim, Actuators.arm,  0.0, timeout=6)
    time.sleep(1.0)
    log("¡Completado!")

# ── Grabación de video compuesto 2×2 (todas las cámaras) ─────────────────────
#
#  ┌──────────────────┬──────────────────┐
#  │  Head RGB        │  Head Depth      │
#  │  (detección HSV) │  (colorizado)    │
#  ├──────────────────┼──────────────────┤
#  │  Wrist D405      │  Overhead        │
#  │                  │  (3ra persona)   │
#  └──────────────────┴──────────────────┘

CELL_W, CELL_H = VIDEO_W // 2, VIDEO_H // 2   # 320 × 240 cada celda


def _annotate_head_rgb(frame, blue_px, red_px, depth_blue, depth_red):
    """Dibuja detecciones sobre el frame de cabeza RGB."""
    out = frame.copy()
    for px, color, bgr, depth in [
        (blue_px, "blue", (255, 80,  0), depth_blue),
        (red_px,  "red",  (0,  50, 255), depth_red),
    ]:
        if px:
            cv2.circle(out, px, 14, bgr, 3)
            label = f"{color} {depth:.2f}m" if depth else color
            cv2.putText(out, label, (px[0] + 8, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)
    return out


def _colorize_depth(depth_raw):
    """Convierte imagen de profundidad uint16 a BGR colorizado."""
    if depth_raw is None:
        return np.zeros((D435I_H, D435I_W, 3), dtype=np.uint8)
    norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _safe_frame(cam_data, camera, fallback_shape):
    try:
        return cam_data.get_camera_data(camera).copy()
    except Exception:
        return np.zeros(fallback_shape, dtype=np.uint8)


class VideoRecorder:
    def __init__(self, sim, path):
        self.sim = sim
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))
        self.label = ""
        self._running = False

    def _build_frame(self, cam_data):
        """Construye el frame compuesto 2×2."""

        # ── Head RGB con overlay de detección ────────────────────────────────
        head_rgb = _safe_frame(cam_data, StretchCameras.cam_d435i_rgb,
                               (D435I_H, D435I_W, 3))

        # Detección y profundidad para el overlay
        blue_px = find_cube_pixel(head_rgb, "blue")
        red_px  = find_cube_pixel(head_rgb, "red")

        depth_blue = depth_red = None
        try:
            depth_raw = cam_data.get_camera_data(
                StretchCameras.cam_d435i_depth, auto_rotate=False)
            if blue_px:
                d = depth_sample(depth_raw, blue_px[0], blue_px[1])
                depth_blue = d * D435I_DEPTH_SCALE if d > 0 else None
            if red_px:
                d = depth_sample(depth_raw, red_px[0], red_px[1])
                depth_red = d * D435I_DEPTH_SCALE if d > 0 else None
        except Exception:
            depth_raw = None

        head_ann = _annotate_head_rgb(head_rgb, blue_px, red_px,
                                      depth_blue, depth_red)
        cv2.putText(head_ann, "Head RGB (D435i)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Head Depth colorizado ─────────────────────────────────────────────
        depth_color = _colorize_depth(depth_raw)
        if depth_raw is None:
            try:
                dr = cam_data.get_camera_data(
                    StretchCameras.cam_d435i_depth, auto_rotate=False)
                depth_color = _colorize_depth(dr)
            except Exception:
                pass
        # Marcar pixels detectados también en depth
        for px, bgr in [(blue_px, (255,80,0)), (red_px, (0,50,255))]:
            if px:
                cv2.circle(depth_color, px, 14, bgr, 3)
        cv2.putText(depth_color, "Head Depth (D435i)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Wrist RGB (D405) ──────────────────────────────────────────────────
        wrist = _safe_frame(cam_data, StretchCameras.cam_d405_rgb, (270, 480, 3))
        wx = find_cube_pixel(wrist, "blue") or find_cube_pixel(wrist, "red")
        if wx:
            cv2.circle(wrist, wx, 14, (0, 255, 0), 3)
        cv2.putText(wrist, "Wrist RGB (D405)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Overhead ──────────────────────────────────────────────────────────
        overhead = _safe_frame(cam_data, StretchCameras.cam_overhead,
                               (VIDEO_H, VIDEO_W, 3))
        cv2.putText(overhead, "Overhead", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Ensamblar grilla 2×2 ──────────────────────────────────────────────
        tl = cv2.resize(head_ann,   (CELL_W, CELL_H))
        tr = cv2.resize(depth_color,(CELL_W, CELL_H))
        bl = cv2.resize(wrist,      (CELL_W, CELL_H))
        br = cv2.resize(overhead,   (CELL_W, CELL_H))

        top = np.hstack([tl, tr])
        bot = np.hstack([bl, br])
        grid = np.vstack([top, bot])

        # Label de fase encima
        if self.label:
            cv2.putText(grid, self.label, (10, VIDEO_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        return grid

    def _loop(self):
        interval = 1.0 / VIDEO_FPS
        while self._running and self.sim.is_running():
            t0 = time.perf_counter()
            try:
                cam_data = self.sim.pull_camera_data()
                frame = self._build_frame(cam_data)
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
    print("Cube Stack Demo — Vision-Based")
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL', 'no seteado')}")
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
        rec.label = "Estabilizando..."
        log("Esperando que los cubos caigan...")
        time.sleep(3.0)

        rec.label = "Calibrando brazo..."
        arm_dir, _, dz_per_lift, theta_cal = probe_arm_geometry(sim)

        rec.label = "Pick and place (vision)..."
        pick_and_place(sim, arm_dir, theta_cal, dz_per_lift)

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
