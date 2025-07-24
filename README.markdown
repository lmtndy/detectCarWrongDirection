# Vehicle Lane Violation Detection Project

## Overview
This project implements a vehicle lane violation detection system using the YOLOv11 model from Ultralytics. It processes a video input to detect and track cars, counting the number of vehicles entering and exiting defined regions (lanes) to identify potential lane violations. The system uses OpenCV for video processing, Supervision for annotations, and Shapely for geometric calculations.

## Features
- Detects cars in a video stream using the YOLOv11 model.
- Tracks vehicles across frames and counts entries/exits in multiple defined regions.
- Visualizes regions, bounding boxes, and counts on the video output.
- Saves the processed video with annotations to a file (`output.mp4`).
- Supports real-time display of the processed video.

## Prerequisites
- Python 3.8+
- Required libraries:
  - `opencv-python`
  - `numpy`
  - `torch`
  - `ultralytics`
  - `supervision`
  - `shapely`
- A pre-trained YOLOv11 model file (`best1024.pt`).
- A video file for input (e.g., `cars_test.mp4`).

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```
   pip install opencv-python numpy torch ultralytics supervision shapely
   ```
3. Place the YOLOv11 model file (`best1024.pt`) in the project directory.
4. Ensure the input video file (e.g., `cars_test.mp4`) is accessible at the specified path.

## Usage
1. Update the `video_path` in `main.py` to point to your input video:
   ```python
   video_path = "path/to/your/video.mp4"
   ```
2. Define regions for lane detection in `region_points` within `main.py`. Each region is a list of four `[x, y]` coordinates forming a polygon:
   ```python
   region_points = [
       [[890, 14], [889, 65], [410, 65], [409, 14]],
       [[892, 70], [893, 119], [410, 114], [409, 68]],
       ...
   ]
   ```
3. Run the script:
   ```
   python main.py
   ```
4. The processed video will be saved as `output.mp4`. If `show=True`, a live window will display the annotated video. Press `q` to exit.

## Code Structure
- `main.py`: Main script containing the `MultipleObjectCounter` class and video processing logic.
- `MultipleObjectCounter`: A class that:
  - Initializes the YOLOv11 model and regions.
  - Tracks vehicles using centroids and counts entries/exits in each region.
  - Annotates the video with bounding boxes, region polygons, and counts.
- Dependencies:
  - YOLOv11 for object detection.
  - OpenCV for video handling.
  - Supervision for drawing annotations.
  - Shapely for region-based geometric checks.

## Input
- **Video**: A video file (e.g., `cars_test.mp4`) containing footage of vehicles.
- **Model**: A YOLOv11 model file (`best1024.pt`) for car detection.
- **Regions**: A list of polygons defined by `[x, y]` coordinates to represent lanes.

## Output
- **Video**: An annotated video (`output.mp4`) with:
  - Bounding boxes around detected cars.
  - Labeled regions with entry/exit counts.
  - Visualized polygons for each region.
- **Live Display**: Optional real-time display of the annotated video.

## Configuration
The `MultipleObjectCounter` class accepts the following parameters:
- `model_path`: Path to the YOLOv11 model file.
- `regions`: List of region coordinates for lane detection.
- `class_names`: List of object classes to detect (default: `["car"]`).
- `show`: Boolean to enable/disable live video display (default: `True`).
- `show_in`: Boolean to show entry counts (default: `True`).
- `show_out`: Boolean to show exit counts (default: `True`).
- `line_width`: Thickness of drawn lines and boxes (default: `2`).

## Example
```python
counter = MultipleObjectCounter(
    model_path="best1024.pt",
    regions=region_points,
    class_names=["car"],
    show=True,
    show_in=True,
    show_out=True,
    line_width=2
)
```

## Notes
- Ensure the video file and model file paths are correct to avoid errors.
- The system assumes a single class (`car`) for detection. Modify `class_names` for additional classes.
- Regions should be defined carefully to match the video's resolution and lane positions.
- Performance depends on hardware; a GPU is recommended for faster processing.

## License
This project is licensed under the MIT License.