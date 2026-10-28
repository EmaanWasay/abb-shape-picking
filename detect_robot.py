#!/usr/bin/env python3
"""
Vision server for the ABB shape pick-and-place (runs on the Raspberry Pi).
Group 1: only circle, square and triangle. Hexagon and star are ignored.

  1. Grabs frames from the Basler camera.
  2. Finds the ArUco marker (reference point) and the shapes (YOLO model).
  3. You choose a shape on the web page (or in the window, if a screen is
     connected to the Pi) and press "Send to Robot".
  4. Sends "dx,dy,shapeID,angle" to the robot and waits for "done".

Put these in ONE folder on the Pi:
    detect_robot.py
    best_ncnn_model/         (model.ncnn.param, model.ncnn.bin, metadata.yaml)

Run:
    python3 detect_robot.py

The script prints the web address, e.g. http://192.168.125.207:5001
Start this script BEFORE you run the robot program.
"""

import math
import os
import queue
import socket
import subprocess
import threading
import time

import cv2
import cv2.aruco as aruco
import numpy as np
from flask import Flask, Response, jsonify, request
from pypylon import pylon
from ultralytics import YOLO

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
    HAVE_TK = True
except ImportError:
    HAVE_TK = False


# =============================================================
# SETTINGS
# =============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO model. metadata.yaml says it was exported at 640x640, so keep 640.
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_ncnn_model")
YOLO_CONF = 0.5
YOLO_IMGSZ = 640
# The model also knows hexagon and star, but group 1 ignores them
TARGET_CLASSES = {"circle", "square", "triangle"}

# ArUco marker used as the reference point
ARUCO_REAL_MM = 100              # side length of the printed marker in mm
ARUCO_ID = 0
ARUCO_DICT = aruco.DICT_5X5_50

# Tuning values (same as Detect_v2)
OFFSET_ADDITION = 0              # mm added to |dx| and |dy|
OFFSET_ANGLE_SUBTRACTION = 0     # degrees added to |rotation|
USE_CALIBRATION = True           # True = correct dx, dy with CALIBRATION_M

# Network
SOCKET_HOST = "0.0.0.0"          # listen on every IP the Pi has
SOCKET_PORT = 5000               # must match SocketConnect in RAPID
WEB_PORT = 5001                  # web page: http://<Pi IP>:5001
ROBOT_TIMEOUT_S = 180            # max wait for "done" (includes yellow button)

# Also open a window on the Pi (needs a screen or VNC; ignored over SSH)
SHOW_WINDOW = True

# Camera
EXPOSURE_US = 200000.0

SHAPE_AREA_BOUNDS = {"min": 6000, "max": 90000}

# Real size filter in mm (uses the ArUco scale). The shapes are about 55 mm,
# the black magnets are much smaller, so anything outside this range is ignored.
# The terminal prints each shape's size, so you can adjust these if needed.
MIN_SHAPE_MM = 35
MAX_SHAPE_MM = 90

# Colour thresholds (OpenCV HSV: H 0-179, S 0-255, V 0-255).
# Only 5 colours exist: red, green, blue, black, white.
COLOR_SAT_MIN = 70               # less saturated than this = black or white
BLACK_WHITE_V = 110              # for those: darker = black, brighter = white

SYMMETRY = {
    "circle": 360,
    "square": 90,
    "triangle": 120,
}

# The angle each cutout "wants" the shape at (tune by trial)
PLACE_ORIENTATION = {
    "triangle": 90,
    "square": 0.0,
    "circle": 0.0,
}

# Must match GetTarget() in RAPID
SHAPE_IDS = {
    "triangle": 1,
    "circle": 2,
    "square": 3,
}

CALIBRATION_M = np.array([
    [9.89676661e-01, 8.23330000e-04, -2.71388391e+00],
    [2.01235076e-02, 9.81780104e-01, 5.88128332e-01],
], dtype=np.float32)


# =============================================================
# SHARED STATE (between threads)
# =============================================================
camera = None
converter = None
model = None

frame_lock = threading.Lock()
latest_detected = None           # JPEG bytes of the annotated frame

detection_lock = threading.Lock()
latest_shapes = []
latest_aruco = None

robot_lock = threading.Lock()    # protects robot_conn
send_lock = threading.Lock()     # only one command to the robot at a time
robot_conn = None
robot_connected = False
robot_busy = False
robot_inbox = queue.Queue()      # messages received from the robot


# =============================================================
# SETUP: MODEL AND CAMERA
# =============================================================
def load_model():
    global model
    if not os.path.isdir(YOLO_MODEL_PATH):
        raise SystemExit(
            f"Model folder not found: {YOLO_MODEL_PATH}\n"
            "Put the 'best_ncnn_model' folder in the same folder as this script."
        )
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH, task="segment")
    dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False)   # warm-up
    print("YOLO ready.")


def open_camera():
    global camera, converter
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    except Exception as e:
        raise SystemExit(f"Basler camera not found ({e}). Check the USB cable.")
    camera.Open()
    camera.ExposureTime.SetValue(EXPOSURE_US)
    camera.PixelFormat.SetValue("RGB8")

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment.Value = "MsbAligned"

    print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def close_camera():
    try:
        if camera is not None:
            camera.StopGrabbing()
            camera.Close()
    except Exception:
        pass


# =============================================================
# ARUCO DETECTION
# =============================================================
def make_aruco_detector():
    params = aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 5
    params.minCornerDistanceRate = 0.01
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    return aruco.ArucoDetector(aruco_dict, params)


aruco_detector = make_aruco_detector()


def detect_aruco(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return None

    idx = np.where(ids.flatten() == ARUCO_ID)[0]
    if len(idx) == 0:
        return None

    c = corners[idx[0]][0]
    cx = np.mean(c[:, 0])
    cy = np.mean(c[:, 1])

    dx = c[1][0] - c[0][0]
    dy = c[1][1] - c[0][1]
    angle = np.degrees(np.arctan2(dy, dx))

    edge_lengths = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
    px_per_mm = np.mean(edge_lengths) / ARUCO_REAL_MM

    return {
        "center_px": (cx, cy),
        "angle_deg": angle,
        "px_per_mm": px_per_mm,
        "corners": c,
        "top_corner": c[0],
    }


# =============================================================
# HELPERS
# =============================================================
def get_shape_color(image_bgr, contour):
    """Average colour inside the shape -> red, green, blue, black or white.

    Uses the whole shape (not one pixel), ignoring a thin band at the edges.
    Returns (colour name, (h, s, v)).
    """
    x, y, w, h = cv2.boundingRect(contour)
    crop = image_bgr[y:y + h, x:x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x, -y))
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if cv2.countNonZero(inner) > 0:
        mask = inner

    b, g, r, _ = cv2.mean(crop, mask=mask)
    pixel = np.uint8([[[round(b), round(g), round(r)]]])
    hue, sat, val = (int(c) for c in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])

    if sat < COLOR_SAT_MIN:
        name = "black" if val < BLACK_WHITE_V else "white"
    elif val < 50:
        name = "black"
    elif hue < 25 or hue >= 150:
        name = "red"
    elif hue < 90:
        name = "green"
    else:
        name = "blue"
    return name, (hue, sat, val)


def add_to_magnitude(number, value_to_add):
    """Add value_to_add to |number| and keep the original sign."""
    return math.copysign(abs(number) + value_to_add, number)


def apply_calibration(dx_mm, dy_mm):
    """Correct a raw ArUco offset before sending it to the robot's Offs()."""
    p = np.array([dx_mm, dy_mm, 1.0])
    out = CALIBRATION_M @ p
    return float(out[0]), float(out[1])


def shape_label(i, s):
    return f"[{i + 1}] {s['color']} {s['name']}"


# =============================================================
# ANGLE MEASUREMENT
# =============================================================
def get_triangle_angle(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    pts = approx.reshape(-1, 2).astype(float)

    if len(pts) < 3:
        rect = cv2.minAreaRect(contour)
        return rect[2] % 120

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    corner_angles = []
    for px, py in pts:
        ang = np.degrees(np.arctan2(py - cy, px - cx))
        corner_angles.append(ang % 120)
    return float(np.median(corner_angles))


def get_stable_angle(contour, shape_name):
    if shape_name == "circle":
        return 0.0
    if shape_name == "triangle":
        return get_triangle_angle(contour)

    # square: minAreaRect wrapped to its symmetry
    rect = cv2.minAreaRect(contour)
    return rect[2] % SYMMETRY.get(shape_name, 360)


def compute_place_rotation(shape_name, pick_angle):
    if shape_name == "circle":
        return 0.0

    place_angle = PLACE_ORIENTATION.get(shape_name, 0.0)
    delta = place_angle - pick_angle
    period = SYMMETRY.get(shape_name, 360)

    delta = delta % period
    if delta > period / 2:
        delta -= period          # shortest rotation direction

    return add_to_magnitude(delta, OFFSET_ANGLE_SUBTRACTION)


# =============================================================
# SHAPE DETECTION (YOLO)
# =============================================================
def detect_shapes(image_bgr):
    img_h, img_w = image_bgr.shape[:2]
    shapes = []

    result = model.predict(image_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]
    if result.masks is None:
        return shapes

    for poly, cls_id, score in zip(result.masks.xy,
                                   result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
        name = result.names[int(cls_id)]
        if name not in TARGET_CLASSES:
            continue

        cnt = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        if len(cnt) < 3:
            continue

        area = cv2.contourArea(cnt)
        if area < SHAPE_AREA_BOUNDS["min"] or area > SHAPE_AREA_BOUNDS["max"]:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = min(max(cx, 0), img_w - 1)
        cy = min(max(cy, 0), img_h - 1)

        color, hsv = get_shape_color(image_bgr, cnt)
        (rw, rh) = cv2.minAreaRect(cnt)[1]

        shapes.append({
            "name": name,
            "color": color,
            "hsv": hsv,
            "size_px": max(rw, rh),       # longest side of the rotated box
            "centroid_px": (cx, cy),
            "angle_deg": round(get_stable_angle(cnt, name), 2),
            "contour": cnt,
            "conf": float(score),
        })

    # Fixed order (left to right, then top to bottom) so the list
    # does not jump around between frames
    shapes.sort(key=lambda s: (s["centroid_px"][0] // 50, s["centroid_px"][1]))
    return shapes


def shape_size_mm(shape, aruco_info):
    return shape["size_px"] / aruco_info["px_per_mm"]


def filter_by_size(shapes, aruco_info):
    """Drop things that are too small or too big to be a shape (e.g. magnets)."""
    if aruco_info is None:
        return shapes                     # no scale without the marker
    return [s for s in shapes
            if MIN_SHAPE_MM <= shape_size_mm(s, aruco_info) <= MAX_SHAPE_MM]


# =============================================================
# OFFSET AND ROBOT MESSAGE
# =============================================================
def compute_offset_mm(cx, cy, aruco_info):
    acx, acy = aruco_info["center_px"]
    px_per_mm = aruco_info["px_per_mm"]

    dx_px = cx - acx
    dy_px = cy - acy

    # Rotate into the marker's frame
    angle_rad = np.radians(-aruco_info["angle_deg"])
    dx_rotated = dx_px * np.cos(angle_rad) - dy_px * np.sin(angle_rad)
    dy_rotated = dx_px * np.sin(angle_rad) + dy_px * np.cos(angle_rad)

    dx_mm = dx_rotated / px_per_mm
    dy_mm = dy_rotated / px_per_mm
    return (add_to_magnitude(dx_mm, OFFSET_ADDITION),
            add_to_magnitude(dy_mm, OFFSET_ADDITION))


def build_robot_message(shape, aruco_info):
    """Build 'dx,dy,shapeID,angle' for the robot (camera dy is robot X)."""
    cx, cy = shape["centroid_px"]
    dx_mm, dy_mm = compute_offset_mm(cx, cy, aruco_info)
    if USE_CALIBRATION:
        dx_mm, dy_mm = apply_calibration(dx_mm, dy_mm)

    pick_angle = shape["angle_deg"] - aruco_info["angle_deg"]
    place_rot = compute_place_rotation(shape["name"], pick_angle)
    shape_id = SHAPE_IDS.get(shape["name"], 0)

    return f"{dy_mm:.2f},{dx_mm:.2f},{shape_id},{place_rot:.2f}"


# =============================================================
# DRAW RESULTS ON THE IMAGE
# =============================================================
def annotate_frame(image_bgr, aruco_info, shapes):
    display = image_bgr.copy()

    if aruco_info is not None:
        acx, acy = aruco_info["center_px"]
        px_per_mm = aruco_info["px_per_mm"]

        cv2.circle(display, (int(acx), int(acy)), 8, (255, 0, 255), -1)
        cv2.putText(display, f"ArUco ID:{ARUCO_ID}", (int(acx) + 15, int(acy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(display, f"{px_per_mm:.2f} px/mm", (int(acx) + 15, int(acy) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        c = aruco_info["corners"].astype(int)
        for j in range(4):
            cv2.line(display, tuple(c[j]), tuple(c[(j + 1) % 4]), (255, 0, 255), 2)

        for i, s in enumerate(shapes):
            cx, cy = s["centroid_px"]
            dx_mm, dy_mm = compute_offset_mm(cx, cy, aruco_info)
            pick_angle = s["angle_deg"] - aruco_info["angle_deg"]
            place_rot = compute_place_rotation(s["name"], pick_angle)
            dist_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2)

            cv2.drawContours(display, [s["contour"]], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(s["contour"])
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(display, (int(acx), int(acy)), (cx, cy), (255, 0, 255), 1)

            cv2.putText(display, shape_label(i, s), (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display, f"dx={dx_mm:.1f} dy={dy_mm:.1f}mm", (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(display, f"dist={dist_mm:.1f}mm", (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(display, f"rot={place_rot:.1f}deg", (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        status = (f"Shapes: {len(shapes)} | Scale: {px_per_mm:.2f} px/mm | "
                  f"Marker: {ARUCO_REAL_MM:.0f}mm")
    else:
        cv2.putText(display, "ArUco NOT DETECTED", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        for s in shapes:
            cx, cy = s["centroid_px"]
            cv2.drawContours(display, [s["contour"]], -1, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        status = f"Shapes: {len(shapes)} | ArUco: NOT FOUND"

    cv2.putText(display, status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return display


def print_detections(shapes, aruco_info):
    if aruco_info is None:
        print(f"ArUco NOT found | shapes: {len(shapes)}")
        return
    print(f"ArUco found | scale {aruco_info['px_per_mm']:.2f} px/mm | shapes: {len(shapes)}")
    for i, s in enumerate(shapes):
        h, sat, v = s["hsv"]
        print(f"   {shape_label(i, s)} | size {shape_size_mm(s, aruco_info):.0f} mm"
              f" | HSV {h},{sat},{v} -> robot message: {build_robot_message(s, aruco_info)}")


# =============================================================
# CAMERA LOOP (runs in its own thread)
# =============================================================
def capture_loop():
    global latest_detected, latest_shapes, latest_aruco
    last_print = 0.0

    while camera.IsGrabbing():
        try:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                continue

            converted = converter.Convert(grab_result)
            image = converted.GetArray()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            grab_result.Release()

            aruco_info = detect_aruco(image_bgr)
            shapes = filter_by_size(detect_shapes(image_bgr), aruco_info)
            display = annotate_frame(image_bgr, aruco_info, shapes)

            with detection_lock:
                latest_shapes = shapes
                latest_aruco = aruco_info

            display = cv2.resize(display, (960, 800))
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with frame_lock:
                    latest_detected = buf.tobytes()

            if time.time() - last_print > 2.0:
                last_print = time.time()
                print_detections(shapes, aruco_info)

        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(0.1)


# =============================================================
# ROBOT CONNECTION (the robot is the client, this script is the server)
# =============================================================
def robot_accept_loop():
    """Wait for the robot, then read everything it sends.

    This thread is the ONLY place that reads from the robot socket.
    Received messages go into robot_inbox.
    """
    global robot_conn, robot_connected

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((SOCKET_HOST, SOCKET_PORT))
    except OSError as e:
        print(f"\n*** Cannot open port {SOCKET_PORT}: {e}")
        print("*** Is this script already running? Close it and start again.\n")
        return
    server.listen(1)
    print(f"Robot server listening on port {SOCKET_PORT}")

    while True:
        print("Waiting for robot to connect...")
        conn, addr = server.accept()
        conn.settimeout(None)

        # Forget messages from an old connection
        while not robot_inbox.empty():
            robot_inbox.get_nowait()

        with robot_lock:
            robot_conn = conn
            robot_connected = True
        print(f"Robot connected from {addr[0]}")

        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break                       # robot closed the connection
                msg = data.decode("latin-1").strip()
                if msg:
                    print(f"Robot says: {msg}")
                    robot_inbox.put(msg)
        except OSError as e:
            print(f"Robot connection error: {e}")
        finally:
            with robot_lock:
                robot_conn = None
                robot_connected = False
            try:
                conn.close()
            except OSError:
                pass
            robot_inbox.put(None)               # wake up anyone waiting for "done"
            print("Robot disconnected.")


def send_to_robot_and_wait(message, timeout=ROBOT_TIMEOUT_S):
    """Send 'dx,dy,shapeID,angle' and wait until the robot answers 'done'."""
    global robot_busy

    if not send_lock.acquire(blocking=False):
        return {"ok": False, "error": "Robot is busy"}
    try:
        with robot_lock:
            conn = robot_conn
        if conn is None:
            return {"ok": False, "error": "Robot not connected"}

        robot_busy = True

        # Throw away old messages (like "ready") so we only react to new ones
        while not robot_inbox.empty():
            robot_inbox.get_nowait()

        conn.sendall(message.encode("ascii"))
        print(f"Sent to robot: {message}")

        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return {"ok": False, "error": "Timeout waiting for robot"}
            try:
                msg = robot_inbox.get(timeout=remaining)
            except queue.Empty:
                return {"ok": False, "error": "Timeout waiting for robot"}

            if msg is None:
                return {"ok": False, "error": "Robot disconnected"}
            if "done" in msg:
                return {"ok": True, "response": "done"}
            if "error" in msg:
                return {"ok": False, "error": "Robot could not read the message"}
            # "ready" or anything else: keep waiting

    except OSError as e:
        return {"ok": False, "error": str(e)}
    finally:
        robot_busy = False
        send_lock.release()


def send_quit():
    """Tell the robot to go home and stop the program."""
    if not send_lock.acquire(blocking=False):
        return {"ok": False, "error": "Robot is busy"}
    try:
        with robot_lock:
            conn = robot_conn
        if conn is None:
            return {"ok": False, "error": "Robot not connected"}
        conn.sendall(b"quit")
        print("Sent to robot: quit")
        return {"ok": True}
    except OSError as e:
        return {"ok": False, "error": str(e)}
    finally:
        send_lock.release()


# =============================================================
# WEB PAGE (open http://<Pi IP>:5001 on the PC)
# =============================================================
app = Flask(__name__)


def generate_detected():
    while True:
        with frame_lock:
            frame = latest_detected
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/detection_feed")
def detection_feed():
    return Response(generate_detected(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/shapes")
def api_shapes():
    with detection_lock:
        shapes = list(latest_shapes)
        aruco_info = latest_aruco
    return jsonify({
        "shapes": [{"index": i, "label": shape_label(i, s)} for i, s in enumerate(shapes)],
        "aruco_found": aruco_info is not None,
        "robot_connected": robot_connected,
        "robot_busy": robot_busy,
    })


@app.route("/api/send", methods=["POST"])
def api_send():
    data = request.get_json(silent=True) or {}
    idx = data.get("index")
    label = data.get("label")

    with detection_lock:
        shapes = list(latest_shapes)
        aruco_info = latest_aruco

    if aruco_info is None:
        return jsonify({"ok": False, "error": "ArUco not detected"})
    if not isinstance(idx, int) or idx < 0 or idx >= len(shapes):
        return jsonify({"ok": False, "error": "Invalid shape selection"})

    shape = shapes[idx]
    if label and label != shape_label(idx, shape):
        return jsonify({"ok": False, "error": "Shapes changed - select again"})

    message = build_robot_message(shape, aruco_info)
    result = send_to_robot_and_wait(message)
    result["sent"] = message
    return jsonify(result)


@app.route("/api/quit", methods=["POST"])
def api_quit():
    return jsonify(send_quit())


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shape Detection & Robot Control</title>
<style>
  body { font-family: Arial, sans-serif; background:#1a1a1a; color:#fff;
         text-align:center; margin:0; padding:20px; }
  h1 { color:#4CAF50; }
  .feed { background:#2a2a2a; padding:10px; border-radius:8px; display:inline-block; }
  img { border-radius:4px; max-width:100%; }
  .controls { margin-top:20px; background:#2a2a2a; padding:20px; border-radius:8px;
              display:inline-block; min-width:300px; }
  select, button { font-size:16px; padding:10px; margin:8px; border-radius:6px; border:none; }
  select { min-width:220px; }
  button { background:#4CAF50; color:white; cursor:pointer; padding:10px 30px; }
  button.quit { background:#c0392b; }
  button:disabled { background:#555; cursor:not-allowed; }
  #status { margin-top:15px; font-size:15px; min-height:24px; }
  .connected { color:#4CAF50; }
  .disconnected { color:#e74c3c; }
  .busy { color:#f39c12; }
</style>
</head>
<body>
<h1>Shape Detection & Robot Control</h1>
<div class="feed"><img src="/detection_feed" width="640"></div>
<br>
<div class="controls">
  <div>
    <label>Shape to pick: </label>
    <select id="shapeSelect"><option>Loading...</option></select>
  </div>
  <div>
    <button id="sendBtn" onclick="sendShape()">Send to Robot</button>
    <button id="quitBtn" class="quit" onclick="quitRobot()">Robot home (quit)</button>
  </div>
  <div id="arucoStatus"></div>
  <div id="robotStatus"></div>
  <div id="status"></div>
</div>
<script>
let locked = false;

async function refreshShapes() {
  const sel = document.getElementById('shapeSelect');
  if (locked || document.activeElement === sel) return;
  try {
    const res = await fetch('/api/shapes');
    const data = await res.json();

    const rs = document.getElementById('robotStatus');
    if (data.robot_busy) {
      rs.innerHTML = '<span class="busy">Robot: BUSY</span>';
    } else if (data.robot_connected) {
      rs.innerHTML = '<span class="connected">Robot: Connected</span>';
    } else {
      rs.innerHTML = '<span class="disconnected">Robot: Not connected</span>';
    }
    document.getElementById('arucoStatus').innerHTML = data.aruco_found
      ? '<span class="connected">ArUco: found</span>'
      : '<span class="disconnected">ArUco: NOT found</span>';

    const prev = sel.value;
    sel.innerHTML = '';
    if (data.shapes.length === 0) {
      const opt = document.createElement('option');
      opt.textContent = 'No shapes detected';
      opt.value = '';
      sel.appendChild(opt);
    } else {
      data.shapes.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.index;
        opt.textContent = s.label;
        sel.appendChild(opt);
      });
      if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    }
  } catch (e) {
    console.error(e);
  }
}

async function sendShape() {
  const sel = document.getElementById('shapeSelect');
  if (sel.value === '' || sel.selectedIndex < 0) {
    setStatus('No shape selected', 'disconnected');
    return;
  }
  const label = sel.options[sel.selectedIndex].textContent;
  locked = true;
  document.getElementById('sendBtn').disabled = true;
  setStatus('Sent ' + label + '. Press the YELLOW button on the robot, then wait...', 'busy');
  try {
    const res = await fetch('/api/send', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index: parseInt(sel.value), label: label})
    });
    const data = await res.json();
    if (data.ok) {
      setStatus('Done! Sent: ' + data.sent + ' | Robot: ' + data.response, 'connected');
    } else {
      setStatus('Error: ' + data.error, 'disconnected');
    }
  } catch (e) {
    setStatus('Request failed: ' + e, 'disconnected');
  } finally {
    locked = false;
    document.getElementById('sendBtn').disabled = false;
  }
}

async function quitRobot() {
  if (!confirm('Send the robot home and stop the robot program?')) return;
  const res = await fetch('/api/quit', {method: 'POST'});
  const data = await res.json();
  setStatus(data.ok ? 'Robot told to quit.' : 'Error: ' + data.error,
            data.ok ? 'connected' : 'disconnected');
}

function setStatus(msg, cls) {
  const s = document.getElementById('status');
  s.className = cls;
  s.textContent = msg;
}

setInterval(refreshShapes, 1000);
refreshShapes();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


def run_web():
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)


# =============================================================
# WINDOW ON THE PI (only if a screen is connected)
# =============================================================
class DetectionGUI:
    def __init__(self, root):
        self.root = root
        root.title("Shape Detection & Robot Control")
        root.configure(bg="#1a1a1a")

        tk.Label(root, text="Shape Detection & Robot Control",
                 font=("Arial", 16, "bold"), fg="#4CAF50", bg="#1a1a1a").pack(pady=8)

        self.video_label = tk.Label(root, bg="#000000")
        self.video_label.pack(padx=10)

        controls = tk.Frame(root, bg="#2a2a2a")
        controls.pack(pady=10, fill="x", padx=10)

        tk.Label(controls, text="Shape to pick:", fg="#fff", bg="#2a2a2a",
                 font=("Arial", 12)).pack(side="left", padx=6)

        self.shape_var = tk.StringVar()
        self.shape_combo = ttk.Combobox(controls, textvariable=self.shape_var,
                                        state="readonly", width=24)
        self.shape_combo.pack(side="left", padx=6)

        self.send_btn = tk.Button(controls, text="Send to Robot", command=self.on_send,
                                  bg="#4CAF50", fg="white", font=("Arial", 12), padx=16)
        self.send_btn.pack(side="left", padx=6)

        self.quit_btn = tk.Button(controls, text="Robot home (quit)", command=self.on_quit,
                                  bg="#c0392b", fg="white", font=("Arial", 12), padx=16)
        self.quit_btn.pack(side="left", padx=6)

        self.robot_label = tk.Label(root, text="", fg="#aaa", bg="#1a1a1a",
                                    font=("Arial", 11))
        self.robot_label.pack()

        self.status_label = tk.Label(root, text="", fg="#aaa", bg="#1a1a1a",
                                     font=("Arial", 11))
        self.status_label.pack(pady=4)

        self.current_shapes = []     # same order as the dropdown
        self.current_aruco = None
        self.locked = False          # True while a pick is running
        self.running = True

        self.update_frame()
        self.update_dropdown()

    def update_frame(self):
        if not self.running:
            return
        with frame_lock:
            jpeg = latest_detected
        if jpeg is not None:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.root.after(50, self.update_frame)

    def update_dropdown(self):
        if not self.running:
            return

        if robot_busy:
            self.robot_label.config(text="Robot: BUSY", fg="#f39c12")
        elif robot_connected:
            self.robot_label.config(text="Robot: Connected", fg="#4CAF50")
        else:
            self.robot_label.config(text="Robot: Not connected", fg="#e74c3c")

        if not self.locked:
            with detection_lock:
                self.current_shapes = list(latest_shapes)
                self.current_aruco = latest_aruco
            labels = [shape_label(i, s) for i, s in enumerate(self.current_shapes)]
            self.shape_combo["values"] = labels
            if labels and self.shape_var.get() not in labels:
                self.shape_combo.current(0)
            elif not labels:
                self.shape_combo.set("No shapes detected")

        self.root.after(1000, self.update_dropdown)

    def on_send(self):
        idx = self.shape_combo.current()
        if idx < 0 or idx >= len(self.current_shapes):
            self.status_label.config(text="No shape selected", fg="#e74c3c")
            return
        if self.current_aruco is None:
            self.status_label.config(text="ArUco not detected", fg="#e74c3c")
            return

        message = build_robot_message(self.current_shapes[idx], self.current_aruco)

        self.locked = True
        self.send_btn.config(state="disabled")
        self.status_label.config(
            text=f"Sent {message}. Press the YELLOW button on the robot, then wait...",
            fg="#f39c12")

        def worker():
            result = send_to_robot_and_wait(message)
            self.root.after(0, lambda: self.finish_send(result, message))

        threading.Thread(target=worker, daemon=True).start()

    def finish_send(self, result, message):
        if result.get("ok"):
            self.status_label.config(
                text=f"Done. Sent {message} | Robot: {result.get('response', '')}",
                fg="#4CAF50")
        else:
            self.status_label.config(text=f"Error: {result.get('error')}", fg="#e74c3c")
        self.locked = False
        self.send_btn.config(state="normal")

    def on_quit(self):
        if not messagebox.askyesno("Quit", "Send the robot home and stop the robot program?"):
            return
        result = send_quit()
        if result.get("ok"):
            self.status_label.config(text="Robot told to quit.", fg="#4CAF50")
        else:
            self.status_label.config(text=f"Error: {result.get('error')}", fg="#e74c3c")

    def on_close(self):
        self.running = False
        self.root.destroy()


def run_window():
    root = tk.Tk()
    root.geometry("1000x1000")
    gui = DetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_close)
    root.mainloop()


# =============================================================
# MAIN
# =============================================================
def print_addresses():
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
        ips = [ip for ip in out.stdout.split() if "." in ip]
    except Exception:
        ips = []

    print("\n" + "=" * 60)
    if ips:
        print("Web page (open on the PC):")
        for ip in ips:
            print(f"   http://{ip}:{WEB_PORT}")
        print("Robot: SocketConnect in RAPID must use the 192.168.125.x one:")
        for ip in ips:
            print(f"   {ip}")
    else:
        print(f"Web page: http://<Pi IP>:{WEB_PORT}  (find the IP with: hostname -I)")
    print("Start the robot program now.")
    print("=" * 60 + "\n")


def main():
    load_model()
    open_camera()

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=robot_accept_loop, daemon=True).start()
    print_addresses()

    try:
        if SHOW_WINDOW and HAVE_TK and os.environ.get("DISPLAY"):
            threading.Thread(target=run_web, daemon=True).start()
            try:
                run_window()
            except Exception as e:
                print(f"Window could not open ({e}). Use the web page. Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        else:
            run_web()          # Ctrl+C to stop
    except KeyboardInterrupt:
        pass
    finally:
        close_camera()
        print("Stopped.")


if __name__ == "__main__":
    main()
