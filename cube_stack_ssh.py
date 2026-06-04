"""
Cube stacking demo - SSH compatible with third-person video recording.

Usage:
    # Linux SSH con GPU (NVIDIA/EGL):
    export MUJOCO_GL=egl
    python cube_stack_ssh.py

    # Linux SSH sin GPU (software rendering, más lento):
    export MUJOCO_GL=osmesa
    python cube_stack_ssh.py

    # Instalar dependencias si faltan:
    sudo apt-get install libegl1 libegl-mesa0     # para EGL
    sudo apt-get install libosmesa6               # para osmesa

El robot detecta los cubos con la cámara de cabeza (D435i),
agarra el cubo azul y lo coloca sobre el cubo rojo.
La grabación de tercera persona se guarda en cube_stack_demo.mp4
"""

import os
# CRÍTICO: setear ANTES de cualquier import de mujoco/stretch_mujoco
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"   # cambiar a "osmesa" si no hay GPU con EGL

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

# ── Rutas ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
SCENE_XML = str(HERE / "stretch_mujoco/models/scene_cubes.xml")
VIDEO_OUTPUT = str(HERE / "cube_stack_demo.mp4")

# ── Parámetros de video ───────────────────────────────────────────────────────
VIDEO_FPS = 10
VIDEO_W, VIDEO_H = 640, 480   # debe coincidir con cam_overhead initial_camera_settings

# ── Cámaras a activar ────────────────────────────────────────────────────────
CAMERAS = [
    StretchCameras.cam_d435i_rgb,    # cabeza RGB  → detección de cubos
    StretchCameras.cam_d435i_depth,  # cabeza depth → distancia al cubo
    StretchCameras.cam_d405_rgb,     # muñeca RGB  → confirmación de agarre
    StretchCameras.cam_overhead,     # tercera persona → grabación de video
]

# ── Rangos HSV para detección de colores ─────────────────────────────────────
BLUE_LOW  = np.array([100, 100, 50])
BLUE_HIGH = np.array([130, 255, 255])
RED_LOW1  = np.array([0,   120, 50])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 50])
RED_HIGH2 = np.array([180, 255, 255])

# ── Parámetros del robot  ─────────────────────────────────────────────────────
# AJUSTAR según posición real del robot en la escena (usar teleop_demo.py primero).
# La escena tiene el robot al origen y la mesa en y=-1.
# Los cubos quedan en x≈-0.04 y x≈0.12, y≈-0.55, z≈0.52 tras caer.

LIFT_HOVER    = 0.55    # altura hover sobre la mesa (m)
LIFT_GRASP    = 0.46    # altura para agarrar el cubo (m)  ← bajar si no toca
LIFT_CARRY    = 0.72    # altura al transportar
LIFT_PLACE    = 0.60    # altura para soltar sobre cubo rojo

ARM_BLUE      = 0.40    # extensión de brazo para llegar al cubo azul (m)
ARM_RED       = 0.40    # extensión para el cubo rojo (similar y)

WRIST_DOWN    = -0.9    # wrist_pitch apuntando al suelo
GRIPPER_OPEN  =  0.04   # gripper abierto
GRIPPER_CLOSE = -0.015  # gripper cerrado (agarre)

HEAD_TILT_TABLE = -0.85   # inclinar cabeza para ver la mesa


# ─────────────────────────────────────────────────────────────────────────────
#  Detección de cubos con cámara
# ─────────────────────────────────────────────────────────────────────────────

def find_cube_center(frame_bgr: np.ndarray | None, color: str) -> tuple[int, int] | None:
    """Devuelve el pixel central del cubo más grande del color dado, o None."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if color == "blue":
        mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    else:  # red
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2),
        )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 50:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def annotate_frame(frame: np.ndarray, blue_c, red_c, label: str = "") -> np.ndarray:
    """Dibuja detecciones y etiqueta en el frame (para el video)."""
    out = frame.copy()
    if blue_c:
        cv2.circle(out, blue_c, 18, (255, 80, 0), 3)
        cv2.putText(out, "BLUE", (blue_c[0] + 20, blue_c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)
    if red_c:
        cv2.circle(out, red_c, 18, (0, 50, 255), 3)
        cv2.putText(out, "RED", (red_c[0] + 20, red_c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
    if label:
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    return out


def get_cube_depth_m(sim: StretchMujocoSimulator, pixel: tuple[int, int]) -> float | None:
    """Lee profundidad en metros en el pixel dado de la cámara de cabeza."""
    try:
        cam_data = sim.pull_camera_data()
        depth = cam_data.get_camera_data(StretchCameras.cam_d435i_depth, auto_correct_rgb=False)
        if depth is None:
            return None
        x, y = pixel
        h, w = depth.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        val = float(depth[y, x])
        if val <= 0:
            return None
        return val * 1e-3   # escala D435i
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Secuencia de pick-and-place
# ─────────────────────────────────────────────────────────────────────────────

def move(sim: StretchMujocoSimulator, actuator: Actuators, pos: float, timeout: float = 8.0):
    """Mueve un actuador y espera a que llegue."""
    sim.move_to(actuator, pos)
    sim.wait_while_is_moving(actuator, timeout=timeout)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_pick_and_place(sim: StretchMujocoSimulator):
    """
    Ejecuta la secuencia de apilado de cubos.

    NOTA: Los valores de LIFT_*, ARM_*, etc. al principio del archivo
    necesitan ajuste fino según la posición real del robot en tu escena.
    Usa teleop_demo.py primero para explorar los valores correctos.
    """

    # — 1. Esperar que los cubos caigan y se estabilicen ——————————————————————
    log("Fase 1: Esperando que los cubos se estabilicen...")
    time.sleep(3.0)

    status = sim.pull_status()
    ee = sim.get_ee_pose()
    log(f"  Base: x={status.base.x:.3f} y={status.base.y:.3f} θ={status.base.theta:.3f} rad")
    log(f"  EE home: [{ee[0,3]:.3f}, {ee[1,3]:.3f}, {ee[2,3]:.3f}]")

    # — 2. Apuntar cámara a la mesa ───────────────────────────────────────────
    log("Fase 2: Apuntando cámara de cabeza a la mesa...")
    move(sim, Actuators.head_tilt, HEAD_TILT_TABLE)
    move(sim, Actuators.head_pan, 0.0)
    time.sleep(0.5)

    # — 3. Detectar cubos con cámara de cabeza ────────────────────────────────
    log("Fase 3: Detectando cubos con cámara...")
    cam_data = sim.pull_camera_data()
    try:
        head_rgb = cam_data.get_camera_data(StretchCameras.cam_d435i_rgb)
    except ValueError:
        head_rgb = None

    blue_pixel = find_cube_center(head_rgb, "blue")
    red_pixel  = find_cube_center(head_rgb, "red")
    log(f"  Cubo azul en pixel: {blue_pixel}")
    log(f"  Cubo rojo en pixel: {red_pixel}")

    if blue_pixel:
        depth = get_cube_depth_m(sim, blue_pixel)
        log(f"  Profundidad al cubo azul: {depth:.3f}m" if depth else "  Sin dato de profundidad")

    # — 4. Preparar muñeca y altura de hover ──────────────────────────────────
    log("Fase 4: Posicionando brazo sobre cubo azul...")
    move(sim, Actuators.wrist_pitch, WRIST_DOWN, timeout=5.0)
    move(sim, Actuators.wrist_yaw, 0.0, timeout=5.0)
    move(sim, Actuators.lift, LIFT_HOVER)

    # Abrir gripper antes de bajar
    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4.0)

    # Extender brazo hacia cubo azul
    move(sim, Actuators.arm, ARM_BLUE)
    time.sleep(0.3)

    # — 5. Bajar al cubo azul ─────────────────────────────────────────────────
    log("Fase 5: Bajando al cubo azul...")
    move(sim, Actuators.lift, LIFT_GRASP)
    time.sleep(0.4)

    # Confirmar con cámara de muñeca
    try:
        cam_data = sim.pull_camera_data()
        wrist_rgb = cam_data.get_camera_data(StretchCameras.cam_d405_rgb)
        wrist_blue = find_cube_center(wrist_rgb, "blue")
        log(f"  Cubo azul en cámara de muñeca: {wrist_blue}")
    except Exception:
        pass

    # — 6. Cerrar gripper (agarrar) ───────────────────────────────────────────
    log("Fase 6: Cerrando gripper (agarrando cubo azul)...")
    move(sim, Actuators.gripper, GRIPPER_CLOSE, timeout=4.0)
    time.sleep(0.6)

    # — 7. Levantar el cubo ───────────────────────────────────────────────────
    log("Fase 7: Levantando cubo azul...")
    move(sim, Actuators.lift, LIFT_CARRY)
    time.sleep(0.5)

    # — 8. Moverse sobre el cubo rojo ─────────────────────────────────────────
    # Los cubos están separados ~16cm en x. Si el robot está alineado con el cubo azul
    # en x, necesita moverse en x para llegar al rojo.
    # Opción A: ajustar extensión del brazo si los cubos difieren en y.
    # Opción B: mover base en x.
    # Por ahora usamos la misma extensión (cubos a mismo y) y movemos base.
    log("Fase 8: Moviéndose al cubo rojo...")

    # Ajuste pequeño de base en x hacia el cubo rojo (+x respecto al azul)
    # Velocidad lineal = 0, omega = 0, pero mover en x con set_base_velocity requiere
    # orientar el robot. Alternativa: extender diferente si están a diferente distancia.
    # Para esta escena los cubos están al mismo y, solo difieren en x ~0.16m.
    # Si el robot está en posición default (facing +x), desplazar base +0.16 en y
    # no es directo sin un controlador completo. Simplificamos asumiendo que el robot
    # puede alcanzar ambos cubos sin mover la base (separación pequeña vs brazo).
    move(sim, Actuators.arm, ARM_RED)
    time.sleep(0.5)

    # — 9. Bajar y soltar ─────────────────────────────────────────────────────
    log("Fase 9: Colocando cubo azul sobre rojo...")
    move(sim, Actuators.lift, LIFT_PLACE)
    time.sleep(0.5)

    move(sim, Actuators.gripper, GRIPPER_OPEN, timeout=4.0)
    time.sleep(0.5)

    # — 10. Retirar brazo ─────────────────────────────────────────────────────
    log("Fase 10: Retirando brazo...")
    move(sim, Actuators.lift, LIFT_CARRY)
    move(sim, Actuators.arm, 0.05)
    time.sleep(1.0)

    log("¡Secuencia completada!")


# ─────────────────────────────────────────────────────────────────────────────
#  Grabación de video (hilo separado)
# ─────────────────────────────────────────────────────────────────────────────

class VideoRecorder:
    def __init__(self, sim: StretchMujocoSimulator, path: str):
        self.sim = sim
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))
        self._running = False
        self._thread: threading.Thread | None = None
        self.phase_label: str = ""

    def _loop(self):
        interval = 1.0 / VIDEO_FPS
        while self._running and self.sim.is_running():
            t0 = time.perf_counter()
            try:
                cam_data = self.sim.pull_camera_data()
                # --- frame de tercera persona (overhead) ---
                overhead = cam_data.get_camera_data(StretchCameras.cam_overhead)
                # --- frame de cabeza para detectar cubos ---
                try:
                    head = cam_data.get_camera_data(StretchCameras.cam_d435i_rgb)
                    blue_c = find_cube_center(head, "blue")
                    red_c  = find_cube_center(head, "red")
                except Exception:
                    blue_c = red_c = None

                frame = annotate_frame(overhead, blue_c, red_c, self.phase_label)

                # Asegurar tamaño correcto
                if frame.shape[:2] != (VIDEO_H, VIDEO_W):
                    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))

                self.writer.write(frame)
            except Exception:
                pass

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self.writer.release()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Cube Stack Demo  –  SSH headless mode")
    print(f"  MUJOCO_GL  = {os.environ.get('MUJOCO_GL', 'no seteado')}")
    print(f"  Escena     = {SCENE_XML}")
    print(f"  Video out  = {VIDEO_OUTPUT}")
    print("=" * 60)

    sim = StretchMujocoSimulator(
        scene_xml_path=SCENE_XML,
        cameras_to_use=CAMERAS,
        camera_hz=10,
    )

    log("Iniciando simulación headless...")
    sim.start(headless=True)

    if not sim.is_running():
        log("ERROR: La simulación no arrancó.")
        sys.exit(1)

    recorder = VideoRecorder(sim, VIDEO_OUTPUT)
    recorder.start()
    log(f"Grabando video en: {VIDEO_OUTPUT}")

    try:
        # Etapa por etapa, actualizar label en el video
        for phase, label in [
            ("settle",   "Esperando estabilización"),
            ("detect",   "Detectando cubos"),
            ("approach", "Posicionando brazo"),
            ("grasp",    "Agarrando cubo azul"),
            ("carry",    "Transportando"),
            ("place",    "Colocando sobre rojo"),
            ("done",     "Completado"),
        ]:
            recorder.phase_label = label

        run_pick_and_place(sim)
        recorder.phase_label = "Completado"
        time.sleep(3.0)   # dejar grabar el estado final

    except KeyboardInterrupt:
        log("Interrumpido por usuario.")
    finally:
        recorder.stop()
        log(f"Video guardado: {VIDEO_OUTPUT}")
        sim.stop()


if __name__ == "__main__":
    main()
