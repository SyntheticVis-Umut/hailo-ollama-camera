# Hailo Detection Image Saver

Simple script to save images when a specific object (e.g., cat) is detected by Hailo NPU.

## Current Status

✅ **IMPLEMENTED**: Option 1 - Using `picamera2` + Hailo SDK directly.

The script now:
- Captures frames using `picamera2` with Raspberry Pi AI camera
- Runs Hailo inference directly using HailoRT SDK
- Parses YOLO detection results from Hailo NPU
- Saves images only when target class is detected with sufficient confidence
- Creates JSON metadata files with detection details

## Implementation Details

### Option 1: Use Hailo SDK Directly (✅ Implemented)

The `save_on_detection.py` script uses:
1. **picamera2** - Captures frames from Raspberry Pi AI camera
2. **Hailo SDK (HailoRT)** - Runs inference directly using VDevice and HEF models
3. **YOLO Detection Parsing** - Parses detection results from Hailo NPU output
4. **Smart Filtering** - Saves images only when target class detected with confidence threshold

### Features

- ✅ Direct access to detection results (class IDs, confidence, bounding boxes)
- ✅ Configurable target class (default: "cat")
- ✅ Configurable confidence threshold (default: 0.7)
- ✅ Automatic model selection (Hailo-8L or Hailo-8)
- ✅ Saves detection metadata as JSON
- ✅ Rate limiting to prevent excessive saves

## Requirements

- Raspberry Pi 5 with Hailo NPU
- Raspberry Pi AI camera module
- Python packages:
  - `picamera2` - Camera interface
  - `hailo_platform` - Hailo SDK
  - `opencv-python` - Image processing
  - `numpy` - Array operations

## Usage

```bash
# Edit the script to set your target class (optional)
nano save_on_detection.py
# Change: TARGET_CLASS = "cat"  # or "dog", "person", etc.
# Change: CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed

# Run the script
python3 save_on_detection.py
```

The script will:
1. Initialize the Hailo NPU and load the YOLOv6 model
2. Initialize the Raspberry Pi camera
3. Continuously capture frames and run inference
4. Save images to `images/` when target class is detected
5. Save detection metadata to `detections/` as JSON files

Press `Ctrl+C` to stop.

## Configuration

Edit these variables in `save_on_detection.py`:

- `TARGET_CLASS` - Object class to detect (e.g., "cat", "dog", "person")
- `CONFIDENCE_THRESHOLD` - Minimum confidence (0.0-1.0) to save image
- `MIN_TIME_BETWEEN_SAVES` - Minimum seconds between saved images

## Files

- `save_on_detection.py` - Main script with picamera2 + Hailo SDK integration
- `config/hailo_config.json` - Hailo detection configuration (for reference)
- `images/` - Saved detection images (created automatically)
- `detections/` - Detection metadata JSON files (created automatically)

## Detection Metadata Format

Each saved image has a corresponding JSON file with:
- Timestamp
- Image path
- Target class information
- All detections (class, confidence, bounding box)
- Target-specific detections filtered by confidence threshold
