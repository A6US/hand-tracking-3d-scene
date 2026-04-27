import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import math

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

rotation = 0.0
gravity = 0.0

# Constantes escena 3D
NUM_POINTS = 500
SCENE_W = 640
SCENE_H = 320
FOV = 400

# Generar puntos base en distribución esférica uniforme
np.random.seed(42)
phi   = np.arccos(2 * np.random.rand(NUM_POINTS) - 1)
theta = 2 * np.pi * np.random.rand(NUM_POINTS)
BASE_RADIUS = np.random.uniform(0.4, 1.0, NUM_POINTS)

base_pts = np.stack([
    BASE_RADIUS * np.sin(phi) * np.cos(theta),
    BASE_RADIUS * np.sin(phi) * np.sin(theta),
    BASE_RADIUS * np.cos(phi)
], axis=1).astype(np.float32)


def get_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def map_value(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def rotation_matrix_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)


def rotation_matrix_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)


def render_3d_scene(rotation_val, gravity_val):
    img = np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)

    # Gravity escala la distancia de los puntos al centro
    scale = map_value(gravity_val, 0, 100, 0.3, 3.0)
    pts = base_pts * scale

    # Rotation gira la cámara alrededor del eje Y
    cam_angle_y = map_value(rotation_val, 0, 100, 0, 2 * math.pi)
    R = rotation_matrix_x(0.3) @ rotation_matrix_y(cam_angle_y)
    rotated = pts @ R.T  # (N, 3)

    # Proyección perspectiva
    cam_z = 4.5
    z = rotated[:, 2] + cam_z
    cx, cy = SCENE_W // 2, SCENE_H // 2
    valid = z > 0.1
    x_proj = np.where(valid, (rotated[:,0] * FOV / z + cx).astype(np.int32), -1)
    y_proj = np.where(valid, (rotated[:,1] * FOV / z + cy).astype(np.int32), -1)
    depth  = np.where(valid, np.clip(1.0 - (z - 1.0) / 7.0, 0.05, 1.0), 0)

    # Pintar de atrás a adelante
    for i in np.argsort(z)[::-1]:
        if not valid[i]:
            continue
        px, py = int(x_proj[i]), int(y_proj[i])
        if not (0 <= px < SCENE_W and 0 <= py < SCENE_H):
            continue
        d = float(depth[i])
        radius = max(1, int(d * 4))
        r = int(255 * d)
        g = int(55 * d * d)
        b = int(25 * d * d)
        if radius >= 3:
            cv2.circle(img, (px,py), radius+2, (int(b*.3),int(g*.3),int(r*.3)), -1, cv2.LINE_AA)
        cv2.circle(img, (px,py), radius, (b,g,r), -1, cv2.LINE_AA)

    # Cuadrícula de referencia tenue
    ov = img.copy()
    cv2.line(ov, (0,cy), (SCENE_W,cy), (35,8,8), 1)
    cv2.line(ov, (cx,0), (cx,SCENE_H), (35,8,8), 1)
    img = cv2.addWeighted(ov, 0.15, img, 0.85, 0)
    cv2.line(img, (0,SCENE_H-1), (SCENE_W,SCENE_H-1), (55,8,8), 1)
    return img

#  TKINTER UI
root = tk.Tk()
root.title("Hand Control · 3D Scene")
root.configure(bg="#080808")
root.resizable(False, False)

outer = tk.Frame(root, bg="#080808")
outer.pack(padx=12, pady=12)

# Escena 3D
scene_canvas = tk.Canvas(outer, width=SCENE_W, height=SCENE_H,
                         bg="black", highlightthickness=1,
                         highlightbackground="#2a0505")
scene_canvas.pack()

tk.Frame(outer, bg="#1a0505", height=3).pack(fill="x")

# Cámara
cam_canvas = tk.Canvas(outer, width=640, height=480,
                       bg="#080808", highlightthickness=0)
cam_canvas.pack()

# Parámetros
params_frame = tk.Frame(outer, bg="#080808")
params_frame.pack(fill="x", pady=(10,0))

FONT_LABEL = ("Courier New", 10, "bold")
FONT_VALUE = ("Courier New", 20, "bold")

def make_param_block(parent, label_text, color, icon):
    f = tk.Frame(parent, bg="#100505", padx=18, pady=10)
    f.pack(side="left", expand=True, fill="both", padx=(0,6))
    top = tk.Frame(f, bg="#100505")
    top.pack(fill="x")
    tk.Label(top, text=icon,         fg=color,   bg="#100505", font=("Courier New",9)).pack(side="left")
    tk.Label(top, text=f"  {label_text}", fg="#664444", bg="#100505", font=FONT_LABEL).pack(side="left")
    val = tk.Label(f, text="000.00", fg=color, bg="#100505", font=FONT_VALUE)
    val.pack(anchor="w", pady=(3,0))
    bar_bg = tk.Frame(f, bg="#1e0808", height=3)
    bar_bg.pack(fill="x", pady=(5,0))
    bar_fg = tk.Frame(bar_bg, bg=color, height=3, width=0)
    bar_fg.place(x=0, y=0, height=3)
    return val, bar_fg, bar_bg

rotation_lbl, rotation_bar, rotation_bg = make_param_block(params_frame, "ROTATION  ·  LEFT HAND",  "#ff3c3c", "↻")
gravity_lbl,  gravity_bar,  gravity_bg  = make_param_block(params_frame, "GRAVITY   ·  RIGHT HAND", "#ff7043", "⊙")

status_bar = tk.Frame(outer, bg="#080808")
status_bar.pack(fill="x", pady=(6,0))
status_lbl = tk.Label(status_bar, text="no hands detected", fg="#331111", bg="#080808", font=("Courier New",9))
status_lbl.pack(side="left")
tk.Label(status_bar, text="[Q] quit", fg="#221111", bg="#080808", font=("Courier New",9)).pack(side="right")


def update_bar(bar_fg, bar_bg, value):
    bar_bg.update_idletasks()
    w = bar_bg.winfo_width()
    bar_fg.place(x=0, y=0, height=3, width=int(value / 100 * w))

#  MediaPipe + Cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DIST_MIN = 20
DIST_MAX = 200
smooth_rotation = 0.0
smooth_gravity  = 0.0
SMOOTH = 0.12


def process_frame():
    global rotation, gravity, smooth_rotation, smooth_gravity

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    status_parts = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            h, w, _ = image.shape
            hand_label = handedness.classification[0].label

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            index_pos = (int(index.x * w), int(index.y * h))

            dist   = get_distance(thumb_pos, index_pos)
            mapped = map_value(dist, DIST_MIN, DIST_MAX, 0, 100)

            if hand_label == "Left":
                rotation = round(mapped, 2)
                color_bgr = (50, 60, 255)
                label_text = f"ROT {rotation:.1f}"
                status_parts.append(f"L→{rotation:.1f}")
            else:
                gravity = round(mapped, 2)
                color_bgr = (60, 120, 255)
                label_text = f"GRV {gravity:.1f}"
                status_parts.append(f"R→{gravity:.1f}")

            mp_drawings.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.line(image, thumb_pos, index_pos, color_bgr, 2)
            cv2.circle(image, thumb_pos, 6, color_bgr, -1)
            cv2.circle(image, index_pos, 6, color_bgr, -1)
            mid = ((thumb_pos[0]+index_pos[0])//2, (thumb_pos[1]+index_pos[1])//2-14)
            cv2.putText(image, label_text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

    # Suavizado (lerp)
    smooth_rotation += (rotation - smooth_rotation) * SMOOTH
    smooth_gravity   += (gravity  - smooth_gravity)  * SMOOTH

    rotation_lbl.config(text=f"{smooth_rotation:06.2f}")
    gravity_lbl.config(text=f"{smooth_gravity:06.2f}")
    update_bar(rotation_bar, rotation_bg, smooth_rotation)
    update_bar(gravity_bar,  gravity_bg,  smooth_gravity)

    if status_parts:
        status_lbl.config(text="  ·  ".join(status_parts), fg="#ff3c3c")
    else:
        status_lbl.config(text="no hands detected", fg="#331111")

    # Escena 3D
    scene_img = render_3d_scene(smooth_rotation, smooth_gravity)
    scene_pil = Image.fromarray(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
    scene_tk  = ImageTk.PhotoImage(scene_pil)
    scene_canvas.img_tk = scene_tk
    scene_canvas.create_image(0, 0, anchor="nw", image=scene_tk)

    # Cámara
    cam_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cam_pil = cam_pil.resize((640, 480))
    cam_tk  = ImageTk.PhotoImage(cam_pil)
    cam_canvas.img_tk = cam_tk
    cam_canvas.create_image(0, 0, anchor="nw", image=cam_tk)

    root.after(10, process_frame)


def on_close():
    cap.release()
    hands_detector.close()
    root.destroy()


root.bind("<q>", lambda e: on_close())
root.bind("<Q>", lambda e: on_close())
root.protocol("WM_DELETE_WINDOW", on_close)

root.after(10, process_frame)
root.mainloop()
