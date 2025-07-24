import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import socket
import json
import threading
import readchar 

# ==============================================================================
# ======================== GLOBAL CONFIGURATION ================================
# ==============================================================================

# --- Network Configuration (Unicast) ---
PORT = 12345
ANDROID_IP = '192.168.137.184' 

# --- Store Configuration ---
# ✅ MODIFICATION: Added the store grid to check for walls. 0 = Path, 1 = Wall.
STORE_GRID = [
    #x= 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # y=0
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=1
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=2
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=3
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=4
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=5
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=6
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # y=7
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=8
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=9
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=10
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=11
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=12
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], # y=13
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # y=14
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], # y=15
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # y=16
]
GRID_WIDTH = len(STORE_GRID[0])
camera_x_pos = 2  # Start at the first valid position

REAL_DISTANCE_START_AISLE_M = 0.65
REAL_DISTANCE_END_AISLE_M   = 6.1
AISLE_Y_START = 1
AISLE_Y_END = 14

# --- Arrow Key Codes ---
LEFT_ARROW = '\x1b[D'
RIGHT_ARROW = '\x1b[C'

# ==============================================================================
# ======================== CALCULATION FUNCTIONS ===============================
# ==============================================================================

def apply_linear_correction(measured_distance):
    """Corrects the measured distance using a linear model y = mx + b."""
    m = (6.0 - 1.0) / (4.6 - 1.0)
    b = 1.0 - m * 1.0
    return m * measured_distance + b

def calculate_grid_position(corrected_distance_m):
    """Calculates the y-coordinate on the grid from the corrected real distance."""
    if REAL_DISTANCE_END_AISLE_M <= REAL_DISTANCE_START_AISLE_M:
        return None
    proportion = (corrected_distance_m - REAL_DISTANCE_START_AISLE_M) / (REAL_DISTANCE_END_AISLE_M - REAL_DISTANCE_START_AISLE_M)
    pos_y = AISLE_Y_START + proportion * (AISLE_Y_END - AISLE_Y_START)
    pos_y_rounded = int(round(pos_y))
    pos_y_final = max(AISLE_Y_START, min(pos_y_rounded, AISLE_Y_END))
    return {"x": camera_x_pos, "y": pos_y_final}

# ✅ MODIFICATION: The logic inside this function is completely new.
def handle_arrow_keys():
    """Handles arrow keys to move the camera, jumping over walls."""
    global camera_x_pos
    
    while True:
        key = readchar.readkey()
        
        if key == RIGHT_ARROW:
            # Default next position is one step to the right.
            next_pos = camera_x_pos + 1
            
            # Check if the next position is a wall (using y=1 as a reference row).
            if next_pos < GRID_WIDTH and STORE_GRID[1][next_pos] == 1:
                print("Wall detected -> Jumping +2")
                next_pos = camera_x_pos + 2 # If it's a wall, jump by two.
            
            # Update position only if it's within the grid boundaries.
            if next_pos < GRID_WIDTH:
                camera_x_pos = next_pos
                print(f"\n➡️  Camera moved to aisle X = {camera_x_pos}")

        elif key == LEFT_ARROW:
            # Default next position is one step to the left.
            next_pos = camera_x_pos - 1

            # Check if the next position is a wall.
            if next_pos >= 0 and STORE_GRID[1][next_pos] == 1:
                print("Wall detected -> Jumping -2")
                next_pos = camera_x_pos - 2 # If it's a wall, jump by two.
            
            # Update position only if it's within the grid boundaries.
            if next_pos >= 0:
                camera_x_pos = next_pos
                print(f"\n⬅️  Camera moved to aisle X = {camera_x_pos}")
                
        elif key == '\x03':
            break

# ==============================================================================
# ======================== INITIALIZATION ======================================
# ==============================================================================

# --- Network, Model, and Camera Initialization ---
print("Initializing UDP server...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Server started. Sending directly to {ANDROID_IP}:{PORT}...")

print("Loading YOLO model...")
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Model loaded.")


print("Starting camera pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
print("Pipeline started.")

# --- Start the key listener thread ---
input_thread = threading.Thread(target=handle_arrow_keys, daemon=True)
input_thread.start()
print("✅ Arrow key listener is active. Make sure this terminal is in focus.")

# ==============================================================================
# ======================== MAIN LOOP ===========================================
# ==============================================================================
last_broadcast_time = 0
try:
    print(f"\nStarting localization and broadcasting. Current aisle X = {camera_x_pos}. Press Ctrl+C to stop.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        current_time = time.time()
        if (current_time - last_broadcast_time) < 0.5:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        height, width, _ = color_image.shape

        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences = [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.6 and classes[class_id] == "person":
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h])
                    confidences.append(float(scores[class_id]))
        
        min_perceived_distance = float('inf')
        if len(boxes) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if isinstance(indexes, np.ndarray):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    roi_x, roi_y = max(0, int(x + w * 0.2)), max(0, int(y + h * 0.2))
                    roi_w, roi_h = int(w * 0.6), int(h * 0.6)
                    depth_roi = depth_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                    valid_depths = depth_roi[depth_roi > 0]
                    if valid_depths.size > 0:
                        distance = np.median(valid_depths) / 1000.0
                        if 0 < distance < min_perceived_distance:
                            min_perceived_distance = distance
            
            if min_perceived_distance != float('inf'):
                real_distance = apply_linear_correction(min_perceived_distance)
                location_data = calculate_grid_position(real_distance)
                
                if location_data:
                    message = json.dumps(location_data).encode('utf-8')
                    sock.sendto(message, (ANDROID_IP, PORT))
                    print(f"Position sent to {ANDROID_IP}: {location_data}")
                
                last_broadcast_time = current_time

except KeyboardInterrupt:
    print("\nShutdown requested by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Stopping camera pipeline.")
    pipeline.stop()
    sock.close()
    print("Program terminated.")