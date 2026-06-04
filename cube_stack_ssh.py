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

BLUE_LO, BLUE_HI = np.array([100, 100, 50]), np.array([130, 255, 255])
RED_LO1, RED_HI1 = np.array([  0, 120, 50]), np.array([ 10, 255, 255])
RED_LO2, RED_HI2 = np.array([170, 120, 50]), np.array([180, 255, 255])

def find_cube_pixel(frame_bgr, color):
    """Detecta el cubo por color y devuelve pixel central (u,v) o None."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
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
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

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
    try:
        cam_data = sim.pull_camera_data()

        # Imagen RGB sin rotar (raw), convertir de RGB a BGR para OpenCV
        rgb_raw = cam_data.get_camera_data(
            StretchCameras.cam_d435i_rgb,
            auto_rotate=False, auto_correct_rgb=False)
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

        # Imagen de profundidad sin rotar
        depth_raw = cam_data.get_camera_data(
            StretchCameras.cam_d435i_depth,
            auto_rotate=False, auto_correct_rgb=False)

    except Exception as e:
        log(f"  Error leyendo cámara: {e}")
        return fallback_xyz.copy()

    # Detectar pixel del cubo
    pixel = find_cube_pixel(bgr_raw, color)
    if pixel is None:
        log(f"  Cubo {color} NO detectado — usando posición de fallback")
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
    """
    Lee la cámara de muñeca (D405) y verifica si el cubo está visible.
    Devuelve pixel central si se detecta, None si no.
    """
    try:
        cam_data = sim.pull_camera_data()
        wrist_rgb = cam_data.get_camera_data(StretchCameras.cam_d405_rgb)
        pixel = find_cube_pixel(wrist_rgb, color)
        if pixel:
            h, w = wrist_rgb.shape[:2]
            norm = (pixel[0] / w - 0.5, pixel[1] / h - 0.5)
            log(f"  Cámara muñeca: cubo {color} en pixel {pixel}, "
                f"desviación del centro: ({norm[0]:.2f}, {norm[1]:.2f})")
        else:
            log(f"  Cámara muñeca: cubo {color} NO visible")
        return pixel
    except Exception as e:
        log(f"  Error cámara muñeca: {e}")
        return None

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

    # ── 5. Posicionar base frente al cubo azul ────────────────────────────────
    log("Fase 4: posicionando base frente al cubo azul...")
    adir_now = current_arm_dir(sim, arm_dir_cal, theta_cal)
    base_tgt = cube_blue[:2] - adir_now * DESIRED_ARM
    move_base_to(sim, base_tgt[0], base_tgt[1])
    ee_ref, adir_now = refresh_refs()

    # Re-detectar con cámara desde nueva posición
    log("  Re-detectando cubo azul desde nueva posición...")
    move(sim, Actuators.head_tilt, -0.85, timeout=4)
    time.sleep(0.4)
    cube_blue2 = detect_cube_3d(sim, "blue", cube_blue)
    if np.linalg.norm(cube_blue2 - cube_blue) < 0.30:
        cube_blue = cube_blue2   # actualizar solo si parece razonable
        log(f"  Posición azul refinada: {np.round(cube_blue, 3)}")

    # ── 6. Hover sobre cubo azul ──────────────────────────────────────────────
    log("Fase 5: hovering sobre cubo azul...")
    lift_h, arm_h = joints_for(cube_blue, ee_ref, adir_now, dz_offset=0.08)
    log(f"  lift={lift_h:.3f}  arm={arm_h:.3f}")
    move(sim, Actuators.lift, lift_h, timeout=8)
    move(sim, Actuators.arm,  arm_h,  timeout=8)
    time.sleep(0.5)
    log(f"  EE real: {np.round(ee_pos(sim), 3)}")

    # ── 7. Verificar con cámara de muñeca (D405) ──────────────────────────────
    log("Fase 6: verificando con cámara de muñeca D405...")
    wrist_px = check_wrist_cam(sim, "blue")

    # ── 8. Bajar al cubo y agarrar ────────────────────────────────────────────
    log("Fase 7: bajando al cubo azul...")
    lift_g, arm_g = joints_for(cube_blue, ee_ref, adir_now, dz_offset=-0.01)
    move(sim, Actuators.lift, lift_g, timeout=6)
    time.sleep(0.5)

    log("Fase 7b: cerrando gripper...")
    move(sim, Actuators.gripper, GRIPPER_CLOSE, timeout=4)
    time.sleep(0.6)

    # Verificar agarre con muñeca
    check_wrist_cam(sim, "blue")

    # ── 9. Levantar cubo ──────────────────────────────────────────────────────
    log("Fase 8: levantando cubo azul...")
    lift_carry = float(np.clip(lift_g + 0.20, 0.05, 1.0))
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

# ── Grabación de video (overhead) ────────────────────────────────────────────

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
                frame = cam.get_camera_data(StretchCameras.cam_overhead).copy()

                # Overlay: detección de cubos en frame de cabeza
                try:
                    head = cam.get_camera_data(StretchCameras.cam_d435i_rgb)
                    for color, bgr in [("blue",(255,80,0)), ("red",(0,50,255))]:
                        px = find_cube_pixel(head, color)
                        if px:
                            cv2.circle(frame, px, 12, bgr, 3)
                except Exception:
                    pass

                if self.label:
                    cv2.putText(frame, self.label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

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
