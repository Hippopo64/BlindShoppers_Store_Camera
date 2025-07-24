import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# --- Mathematical correction function ---
def apply_linear_correction(measured_distance):
    """
    Corrects the distance using a linear function (y = mx + b)
    calibrated on two known points.
    - Point 1 (x1, y1): (measured=1.0, real=1.0)
    - Point 2 (x2, y2): (measured=4.6, real=6.0)
    """
    # Calculate the slope (m)
    m = (6.0 - 1.0) / (4.6 - 1.0)
    
    # Calculate the y-intercept (b) using point 1
    # y = mx + b  =>  b = y - mx
    b = 1.0 - m * 1.0
    
    # Apply the correction
    corrected_distance = m * measured_distance + b
    
    # Safety check to prevent the correction from reducing the distance at close range
    return max(measured_distance, corrected_distance)


# --- Path configuration (recommended) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Make sure the paths are correct for your setup
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
align_to = rs.stream.color
align = rs.align(align_to)
print("Pipeline started.")

# --- Main loop ---
last_print_time = 0
try:
    print("\nStarting detection. Press Ctrl+C to stop.")
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
        height, width, channels = color_image.shape

        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        if len(boxes) > 0:
            min_perceived_distance = float('inf')
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if isinstance(indexes, np.ndarray):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    roi_x = max(0, int(x + w * 0.2))
                    roi_y = max(0, int(y + h * 0.2))
                    roi_w = int(w * 0.6)
                    roi_h = int(h * 0.6)
                    roi_x_end = min(width, roi_x + roi_w)
                    roi_y_end = min(height, roi_y + roi_h)

                    depth_roi = depth_image[roi_y:roi_y_end, roi_x:roi_x_end]
                    valid_depths = depth_roi[depth_roi > 0]
                    
                    if valid_depths.size > 0:
                        distance = np.median(valid_depths) / 1000.0
                        
                        if 0 < distance < min_perceived_distance:
                            min_perceived_distance = distance
            
            if min_perceived_distance != float('inf'):
                # ✅ APPLY THE CORRECTION
                corrected_distance = apply_linear_correction(min_perceived_distance)
                
                # ✅ DISPLAY BOTH VALUES
                print(f"Perceived distance: {min_perceived_distance:.2f}m -> Corrected distance: {corrected_distance:.2f}m")
                last_print_time = current_time

except KeyboardInterrupt:
    print("\nShutdown requested by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Stopping the camera pipeline.")
    pipeline.stop()