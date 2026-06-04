"""
Genera los PNG de marcadores ArUco para los cubos.
Ejecutar UNA vez antes de correr cube_stack_ssh.py:
    python generate_aruco.py
"""
from pathlib import Path
import numpy as np
import cv2

OUT_DIR = Path(__file__).parent / "stretch_mujoco/models"
DICT_ID = cv2.aruco.DICT_4X4_50
SIZE_PX = 256   # resolución del PNG

def make_marker(aruco_id: int, out_path: Path):
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    # API compatible con OpenCV ≥ 4.7
    img = cv2.aruco.generateImageMarker(d, aruco_id, SIZE_PX)
    # Añadir borde blanco (necesario para detección fiable)
    border = SIZE_PX // 8
    canvas = np.ones((SIZE_PX + 2*border, SIZE_PX + 2*border), dtype=np.uint8) * 255
    canvas[border:border+SIZE_PX, border:border+SIZE_PX] = img
    cv2.imwrite(str(out_path), canvas)
    print(f"  Guardado: {out_path}")

if __name__ == "__main__":
    print("Generando marcadores ArUco (DICT_4X4_50)...")
    make_marker(0, OUT_DIR / "aruco_0.png")   # cubo azul
    make_marker(1, OUT_DIR / "aruco_1.png")   # cubo rojo
    print("Listo.")
