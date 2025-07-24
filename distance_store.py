import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# ==============================================================================
# ======================== STORE CONFIGURATION =================================
# ==============================================================================

# REAL distance in meters between the camera and reference points on the map.
REAL_DISTANCE_START_AISLE_M = 0.65  # Real distance for y=1
REAL_DISTANCE_END_AISLE_M   = 6.1   # Real distance for y=14

# Coordinates on the map
CAMERA_X_POS = 2
AISLE_Y_START = 1
AISLE_Y_END = 14

# ==============================================================================
# ======================== CALCULATION FUNCTIONS ===============================
# ==============================================================================

def apply_linear_correction(measured_distance):
    """
    Corrects the measured distance using a linear function (y = mx + b)
    calibrated on two known points.
    - Point 1 (x1, y1): (measured=1.0, real=1.0)
    - Point 2 (x2, y2): (measured=4.6, real=6.0)
    """
    # Your calculation, which is perfectly correct:
    m = (6.0 - 1.0) / (4.6 - 1.0)  # Slope
    b = 1.0 - m * 1.0              # Y-intercept
    
    corrected_distance = m * measured_distance + b
    
    # ✅ THE FIX: Always return the result of the calculation.
    return corrected_distance

def calculate_grid_position(corrected_distance_m):
    """Calculates the y-coordinate on the grid from the corrected REAL distance."""
    if REAL_DISTANCE_END_AISLE_M <= REAL_DISTANCE_START_AISLE_M:
        return None

    proportion = (corrected_distance_m - REAL_DISTANCE_START_AISLE_M) / (REAL_DISTANCE_END_AISLE_M - REAL_DISTANCE_START_AISLE_M)
    pos_y = AISLE_Y_START + proportion * (AISLE_Y_END - AISLE_Y_START)
    
    pos_y_rounded = int(round(pos_y))
    pos_y_final = max(AISLE_Y_START, min(pos_y_rounded, AISLE_Y_END))
    
    return (CAMERA_X_POS, pos_y_final)

# ==============================================================================
# ======================== INITIALIZATION ======================================
# ==============================================================================

# --- Path Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")

# --- YOLO Model Initialization ---
print("Loading YOLO model...")
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

# --- RealSense Camera Initialization ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
print("Starting the camera pipeline...")
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
print("Pipeline started.")

# ==============================================================================
# ======================== MAIN LOOP ===========================================
# ==============================================================================
last_print_time = 0
try:
    print("\nStarting localization. Press Ctrl+C to stop.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        current_time = time.time()
        if (current_time - last_print_time) < 0.5:
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
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
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
                position = calculate_grid_position(real_distance)
                
                if position:
                    print(f"Position: (x={position[0]}, y={position[1]}) | Corrected Distance: {real_distance:.2f}m")
                
                last_print_time = current_time

except KeyboardInterrupt:
    print("\nShutdown requested by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Stopping the camera pipeline.")
    pipeline.stop()