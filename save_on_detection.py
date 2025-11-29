#!/usr/bin/env python3
"""
Hailo Detection Image Saver - Option 1 Implementation
Uses picamera2 + Hailo SDK directly to capture frames and run inference.
Saves images only when target object (person) is detected by Hailo NPU.
"""

import json
import time
import threading
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string
import socket

# Hailo SDK imports
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: Hailo SDK not available. Install hailo_platform package.")

# Configuration
PROJECT_DIR = Path(__file__).parent
IMAGE_DIR = PROJECT_DIR / "images"
DETECTION_DIR = PROJECT_DIR / "detections"

# Detection settings
TARGET_CLASS = "person"  # COCO class ID for person is 0 (YOLOv6 uses COCO classes)
CONFIDENCE_THRESHOLD = 0.7
MIN_TIME_BETWEEN_SAVES = 5.0  # seconds
MODEL_INPUT_SIZE = 640  # YOLOv6 input size

# Web server settings
WEB_PORT = 3009
# Global frame buffer for web streaming
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()

# HEF model paths (try H8L first, fallback to H8)
HEF_PATH_H8L = "/usr/share/hailo-models/yolov6n_h8l.hef"
HEF_PATH_H8 = "/usr/share/hailo-models/yolov6n_h8.hef"

# COCO class names (YOLOv6 uses COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Get class ID for target
try:
    TARGET_CLASS_ID = COCO_CLASSES.index(TARGET_CLASS)
except ValueError:
    print(f"Warning: '{TARGET_CLASS}' not found in COCO classes. Using class ID 0 (person).")
    TARGET_CLASS_ID = 0  # Default to person

# Ensure directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DETECTION_DIR.mkdir(parents=True, exist_ok=True)


class HailoDetector:
    """Hailo detector using Hailo SDK directly."""
    
    def __init__(self, hef_path: str):
        """Initialize Hailo detector with HEF model."""
        if not HAILO_AVAILABLE:
            raise RuntimeError("Hailo SDK not available")
        
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
        
        self.hef_path = hef_path
        self.vdevice = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.input_shape = None
        self.output_shape = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Hailo device and load model."""
        print(f"Loading HEF model: {self.hef_path}")
        
        # Create VDevice
        self.vdevice = VDevice()
        
        # Load HEF
        hef = HEF(self.hef_path)
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, 
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.vdevice.configure(hef, configure_params)
        self.network_group = self.network_groups[0]
        
        # Get input/output stream info
        input_vstream_infos = hef.get_input_vstream_infos()
        output_vstream_infos = hef.get_output_vstream_infos()
        
        # Create vstream params
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32
        )
        
        # Get input shape (assuming single input)
        if input_vstream_infos:
            self.input_shape = input_vstream_infos[0].shape
            print(f"Model input shape: {self.input_shape}")
        
        # Get output shape (assuming single output with NMS)
        if output_vstream_infos:
            self.output_shape = output_vstream_infos[0].shape
            print(f"Model output shape: {self.output_shape}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input (resize, normalize, convert format)."""
        # Resize to model input size
        resized = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        
        # Convert BGR to RGB (if needed)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to uint8 (YOLO expects uint8)
        # YOLO models typically expect values in [0, 255] range as uint8
        if resized.dtype != np.uint8:
            resized = (resized * 255).astype(np.uint8)
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, axis=0)
        
        return resized
    
    def _add_detection(self, detections: list, class_id: int, x_center: float, y_center: float, 
                      width: float, height: float, confidence: float):
        """Helper method to add a detection to the list."""
        # Convert normalized coordinates to pixel coordinates
        x_min = (x_center - width / 2) * MODEL_INPUT_SIZE
        y_min = (y_center - height / 2) * MODEL_INPUT_SIZE
        x_max = (x_center + width / 2) * MODEL_INPUT_SIZE
        y_max = (y_center + height / 2) * MODEL_INPUT_SIZE
        
        detections.append({
            'class_id': int(class_id),
            'class_name': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}",
            'confidence': float(confidence),
            'bbox': {
                'x_min': float(x_min),
                'y_min': float(y_min),
                'x_max': float(x_max),
                'y_max': float(y_max),
                'width': float(width * MODEL_INPUT_SIZE),
                'height': float(height * MODEL_INPUT_SIZE)
            }
        })
    
    def parse_detections(self, output: dict) -> list:
        """
        Parse YOLO detection output.
        YOLO with NMS output format can be:
        - Nested list structure: [batch][class][detection][x, y, w, h, conf]
        - Or numpy array: (80, 5, 100) = [num_classes, detection_data, max_detections]
        """
        detections = []
        
        # Get the output tensor (assuming single output)
        if not output:
            return detections
        
        # Get first output (YOLO models with NMS typically have one output)
        output_key = list(output.keys())[0]
        output_data = output[output_key]
        
        # Debug: print output structure for troubleshooting
        if not hasattr(self, '_debug_printed'):
            print(f"Debug: Output key: {output_key}")
            print(f"Debug: Output type: {type(output_data)}")
            if isinstance(output_data, np.ndarray):
                print(f"Debug: Output shape: {output_data.shape}, dtype: {output_data.dtype}")
            elif isinstance(output_data, list):
                print(f"Debug: Output is list, length: {len(output_data)}")
                if len(output_data) > 0:
                    print(f"Debug: First element type: {type(output_data[0])}, length: {len(output_data[0]) if hasattr(output_data[0], '__len__') else 'N/A'}")
            self._debug_printed = True
        
        # Handle nested list structure (common with Hailo NMS output)
        if isinstance(output_data, list):
            # Structure: [batch][class][detection] or [class][detection]
            # Remove batch dimension if present (usually size 1)
            if len(output_data) > 0 and isinstance(output_data[0], (list, np.ndarray)):
                # Check if first dimension is batch (usually size 1)
                if len(output_data) == 1:
                    output_data = output_data[0]  # Remove batch dimension
                
                # Now we should have [class][detection] structure (80 classes)
                # Iterate over each class
                for class_id, class_detections in enumerate(output_data):
                    if class_detections is None:
                        continue
                    
                    # Handle as list of detections
                    if isinstance(class_detections, list):
                        # Each element in class_detections is a detection
                        for det in class_detections:
                            if det is None:
                                continue
                            
                            # Try to extract detection values
                            try:
                                if isinstance(det, list):
                                    # List format: [x, y, w, h, conf] or similar
                                    if len(det) >= 5:
                                        det_values = [float(v) for v in det[:5]]
                                        x_center, y_center, width, height, confidence = det_values
                                        if confidence >= 0.01:
                                            self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
                                elif isinstance(det, np.ndarray):
                                    # Numpy array format
                                    if len(det) >= 5:
                                        x_center, y_center, width, height, confidence = det[:5]
                                        if confidence >= 0.01:
                                            self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
                            except (ValueError, TypeError, IndexError):
                                continue
                    
                    # Handle as numpy array
                    elif isinstance(class_detections, np.ndarray):
                        if len(class_detections.shape) == 2:
                            # Shape: (num_detections, 5) or (5, num_detections)
                            if class_detections.shape[0] == 5 and class_detections.shape[1] > 5:
                                # Format: (5, num_detections) - transpose
                                class_detections = class_detections.T
                            
                            # Now should be (num_detections, 5)
                            for det in class_detections:
                                if len(det) >= 5:
                                    x_center, y_center, width, height, confidence = det[:5]
                                    if confidence >= 0.01:
                                        self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
                        elif len(class_detections.shape) == 1 and len(class_detections) >= 5:
                            # Single detection for this class
                            x_center, y_center, width, height, confidence = class_detections[:5]
                            if confidence >= 0.01:
                                self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
        
        # Handle numpy array format directly
        elif isinstance(output_data, np.ndarray):
            # Handle different output shapes
            # Shape (80, 5, 100) could mean: [num_classes, detection_data, max_detections]
            # OR (80, 100, 5) could mean: [num_classes, max_detections, detection_data]
            if len(output_data.shape) == 3:
                num_classes, dim1, dim2 = output_data.shape
            
            # Try format (80, 5, 100): [num_classes, detection_data, max_detections]
            if dim1 == 5 and dim2 == 100:
                # Iterate over each class
                for class_id in range(num_classes):
                    class_detections = output_data[class_id]  # Shape: (5, 100)
                    
                    # Iterate over detections for this class
                    for det_idx in range(dim2):
                        # Get detection data: [x_center, y_center, width, height, confidence]
                        det = class_detections[:, det_idx]  # Shape: (5,)
                        
                        if len(det) >= 5:
                            x_center, y_center, width, height, confidence = det[:5]
                            
                            # Skip low confidence or invalid detections
                            if confidence >= 0.01:
                                self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
            
            # Try format (80, 100, 5): [num_classes, max_detections, detection_data]
            elif dim1 == 100 and dim2 == 5:
                # Iterate over each class
                for class_id in range(num_classes):
                    class_detections = output_data[class_id]  # Shape: (100, 5)
                    
                    # Iterate over detections for this class
                    for det_idx in range(dim1):
                        # Get detection data: [x_center, y_center, width, height, confidence]
                        det = class_detections[det_idx]  # Shape: (5,)
                        
                        if len(det) >= 5:
                            x_center, y_center, width, height, confidence = det[:5]
                            
                            # Skip low confidence or invalid detections
                            if confidence >= 0.01:
                                self._add_detection(detections, class_id, x_center, y_center, width, height, confidence)
            else:
                # Unknown format - try to handle generically
                print(f"Warning: Unknown output format shape {output_data.shape}, attempting generic parsing")
                # Flatten and try to parse
                output_flat = output_data.flatten()
                # This is a fallback - may not work correctly
                pass
        
        # Alternative format: [num_detections, 6] where 6 = [x, y, w, h, conf, class_id]
        elif len(output_data.shape) == 2 and output_data.shape[1] >= 6:
            for det in output_data:
                if len(det) >= 6:
                    x_center, y_center, width, height, confidence, class_id = det[:6]
                    
                    # Skip low confidence or invalid detections
                    if confidence < 0.01:
                        continue
                    
                    class_id = int(class_id)
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_min = (x_center - width / 2) * MODEL_INPUT_SIZE
                    y_min = (y_center - height / 2) * MODEL_INPUT_SIZE
                    x_max = (x_center + width / 2) * MODEL_INPUT_SIZE
                    y_max = (y_center + height / 2) * MODEL_INPUT_SIZE
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                        'confidence': float(confidence),
                        'bbox': {
                            'x_min': float(x_min),
                            'y_min': float(y_min),
                            'x_max': float(x_max),
                            'y_max': float(y_max),
                            'width': float(width * MODEL_INPUT_SIZE),
                            'height': float(height * MODEL_INPUT_SIZE)
                        }
                    })
        
        return detections
    
    def infer(self, frame: np.ndarray) -> list:
        """Run inference on a frame and return detections."""
        # Preprocess frame
        preprocessed = self.preprocess(frame)
        
        # Prepare input dict
        input_vstream_info = self.network_group.get_input_vstream_infos()[0]
        input_name = input_vstream_info.name
        input_data = {input_name: preprocessed}
        
        # Run inference
        network_group_params = self.network_group.create_params()
        
        with InferVStreams(
            self.network_group, 
            self.input_vstreams_params, 
            self.output_vstreams_params
        ) as infer_pipeline:
            with self.network_group.activate(network_group_params):
                output = infer_pipeline.infer(input_data)
        
        # Parse detections
        detections = self.parse_detections(output)
        
        return detections
    
    def cleanup(self):
        """Clean up resources."""
        if self.vdevice:
            # VDevice cleanup is handled automatically, but we can release explicitly
            pass


def check_target_detected(detections: list) -> tuple[bool, list]:
    """Check if target class is in detections with sufficient confidence."""
    target_detections = []
    for det in detections:
        if isinstance(det, dict):
            class_id = det.get('class_id', -1)
            confidence = det.get('confidence', 0.0)
            
            if class_id == TARGET_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                target_detections.append(det)
    
    return len(target_detections) > 0, target_detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame - only for target class (person)."""
    frame_with_boxes = frame.copy()
    
    # Convert RGB to BGR for OpenCV drawing
    if len(frame_with_boxes.shape) == 3 and frame_with_boxes.shape[2] == 3:
        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
    
    # Filter detections to only show target class
    target_detections = [det for det in detections 
                        if isinstance(det, dict) and det.get('class_id', -1) == TARGET_CLASS_ID]
    
    for det in target_detections:
        if isinstance(det, dict):
            bbox = det.get('bbox', {})
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', -1)
            
            x_min = int(bbox.get('x_min', 0))
            y_min = int(bbox.get('y_min', 0))
            x_max = int(bbox.get('x_max', 0))
            y_max = int(bbox.get('y_max', 0))
            
            # Scale coordinates from model input size (640x640) to actual frame size
            frame_h, frame_w = frame_with_boxes.shape[:2]
            scale_x = frame_w / MODEL_INPUT_SIZE
            scale_y = frame_h / MODEL_INPUT_SIZE
            
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)
            
            # Use green color for target class (person)
            color = (0, 255, 0)  # Green for target class
            
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y_min - 10, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame_with_boxes, 
                         (x_min, label_y - label_size[1] - 5),
                         (x_min + label_size[0], label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame_with_boxes, label,
                       (x_min, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_with_boxes


def capture_and_check(picam2: Picamera2, detector: HailoDetector) -> bool:
    """Capture a frame, run inference, and save if target detected."""
    global latest_frame, latest_detections
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = IMAGE_DIR / f"detection_{timestamp}.jpg"
    
    try:
        # Capture frame from camera
        frame = picam2.capture_array("main")
        
        # Run Hailo inference
        detections = detector.infer(frame)
        
        # Update global frame buffer for web streaming
        frame_with_boxes = draw_detections(frame, detections)
        with frame_lock:
            latest_frame = frame_with_boxes
            latest_detections = detections
        
        # Check if target class detected
        detected, target_detections = check_target_detected(detections)
        
        if detected:
            # Save image
            # Convert RGB to BGR for OpenCV (if needed)
            if len(frame.shape) == 3:
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
            else:
                save_frame = frame
            
            cv2.imwrite(str(image_file), save_frame)
            
            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_file),
                "target_class": TARGET_CLASS,
                "target_class_id": TARGET_CLASS_ID,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "detected": True,
                "num_target_detections": len(target_detections),
                "target_detections": [
                    {
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"]
                    }
                    for det in target_detections
                ],
                "all_detections": [
                    {
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"]
                    }
                    for det in detections
                ]
            }
            
            metadata_file = DETECTION_DIR / f"detection_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved: {image_file.name} ({len(target_detections)} {TARGET_CLASS}(s) detected)")
            return True
        else:
            print(f"  (No {TARGET_CLASS} detected - skipped)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo Detection Live Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .video-container {
            text-align: center;
            background-color: #000;
            padding: 10px;
            border-radius: 10px;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        .info {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            margin: 5px;
        }
        .status.active {
            background-color: #4CAF50;
        }
        .status.inactive {
            background-color: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Hailo Detection Live Stream</h1>
        <div class="info">
            <strong>Target Class:</strong> <span class="status active">{{ target_class }}</span>
            <strong>Confidence Threshold:</strong> {{ confidence_threshold }}
            <strong>Status:</strong> <span class="status active">LIVE</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Detection Stream">
        </div>
        <div class="info">
            <p><strong>Instructions:</strong> This page shows live camera feed with YOLO detection bounding boxes.</p>
            <p>Green boxes indicate detected {{ target_class }} objects. Orange boxes show other detected objects.</p>
        </div>
    </div>
</body>
</html>
"""


def get_local_ip():
    """Get the local IP address of the device."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def generate_frames():
    """Generate MJPEG frames for video streaming."""
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                # Create a placeholder frame if no frame available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


def create_flask_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Serve the main HTML page."""
        return render_template_string(HTML_TEMPLATE,
                                    target_class=TARGET_CLASS,
                                    confidence_threshold=CONFIDENCE_THRESHOLD)
    
    @app.route('/video_feed')
    def video_feed():
        """Video streaming route."""
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/api/detections')
    def api_detections():
        """API endpoint to get current detections."""
        with frame_lock:
            return Response(json.dumps(latest_detections, default=str),
                          mimetype='application/json')
    
    return app


def start_web_server():
    """Start Flask web server in a separate thread."""
    app = create_flask_app()
    local_ip = get_local_ip()
    print(f"\n🌐 Web server starting...")
    print(f"   Access at: http://{local_ip}:{WEB_PORT}")
    print(f"   Or locally: http://localhost:{WEB_PORT}")
    print("=" * 60)
    
    # Run Flask in a way that doesn't block
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)


def main():
    """Main loop."""
    print("=" * 60)
    print("Hailo Detection Image Saver - Option 1 (picamera2 + Hailo SDK)")
    print("=" * 60)
    print(f"Target class: {TARGET_CLASS} (class ID: {TARGET_CLASS_ID})")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Images directory: {IMAGE_DIR}")
    print("=" * 60)
    
    if not HAILO_AVAILABLE:
        print("\n✗ ERROR: Hailo SDK not available!")
        print("Please install hailo_platform package.")
        return
    
    # Find HEF file
    hef_path = None
    if Path(HEF_PATH_H8L).exists():
        hef_path = HEF_PATH_H8L
        print(f"\nUsing HEF: {hef_path} (Hailo-8L)")
    elif Path(HEF_PATH_H8).exists():
        hef_path = HEF_PATH_H8
        print(f"\nUsing HEF: {hef_path} (Hailo-8)")
    else:
        print(f"\n✗ ERROR: HEF file not found!")
        print(f"Expected at: {HEF_PATH_H8L} or {HEF_PATH_H8}")
        return
    
    # Initialize Hailo detector
    try:
        detector = HailoDetector(hef_path)
    except Exception as e:
        print(f"\n✗ ERROR initializing Hailo detector: {e}")
        return
    
    # Initialize camera
    print("\nInitializing Raspberry Pi camera...")
    try:
        picam2 = Picamera2()
        
        # Configure camera for fast video streaming
        # Lower resolution for higher FPS (640x480 for 60fps)
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=3  # More buffers for smoother streaming
        )
        picam2.configure(config)
        picam2.start()
        
        # Allow camera to stabilize
        time.sleep(1)
        print("Camera ready! (640x480 @ 60fps)")
        
    except Exception as e:
        print(f"\n✗ ERROR initializing camera: {e}")
        detector.cleanup()
        return
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    time.sleep(1)  # Give web server time to start
    
    print("\nStarting capture loop... (Press Ctrl+C to stop)")
    print("Web interface is running in the background.\n")
    
    last_save_time = 0
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            current_time = time.time()
            
            # Only capture if enough time has passed
            if current_time - last_save_time >= MIN_TIME_BETWEEN_SAVES:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame {frame_count}...", end=" ")
                
                if capture_and_check(picam2, detector):
                    last_save_time = current_time
                
                # Small delay between frames
                time.sleep(0.5)
            else:
                # Wait until we can capture again
                wait_time = MIN_TIME_BETWEEN_SAVES - (current_time - last_save_time)
                time.sleep(min(wait_time, 1.0))
                
    except KeyboardInterrupt:
        print("\n\nStopped.")
        print(f"Total frames processed: {frame_count}")
    finally:
        # Cleanup
        print("Cleaning up...")
        picam2.stop()
        detector.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
