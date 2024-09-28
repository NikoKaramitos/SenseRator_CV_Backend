# SenseRator2.0: YOLOv8 Object Detection and Tracking Project

This project utilizes a custom-trained YOLOv8 model for real-time object detection and tracking with video stream recording. The primary application is detecting various street-level objects (e.g., sidewalks, crosswalks, traffic lights, stop signs, etc.) and calculating safety-related indices to determine a **Pedestrian Flow Safety (PFS) index** based on detected objects.

## Features
- **Object Detection and Tracking:**
  Detects and tracks multiple objects using a YOLOv8 model with ByteTrack for object tracking.
- **Real-time Inference:**
   Processes video streams in real-time using the jetson_utils library for efficient handling of Jetson devices.
- **Score Calculation:**
   Computes safety-related indices based on the detected objects, providing insights into pedestrian safety.
- **Video Recording:**
   Records the video stream with object annotations and saves it to the `videos/` directory in `.mp4` format without overwriting previous recordings.

## Requirements
- **Hardware:**
  - Jetson Nano or compatible device with a CSI camera.
- **Software:**
  - Python 3.8+
  - JetPack 4.6 or newer
  - `ultralytics` library for YOLOv8
  - `jetson_utils` for video capture and display
  - `opencv-python`
  - `numpy`

## Installation

1. **Install Dependencies:**  
   First, make sure you have the necessary Python dependencies:
   ```
    pip install ultralytics opencv-python numpy
   ```
2. **Set Up Jetson Utilities:**  
  Follow [NVIDIA's documentation](https://github.com/dusty-nv/jetson-utils) `(dusty-nv)` to install and set up `jetson_utils` for video capture and display on Jetson devices.

4. **Clone or Download the Project:**
  ```
    git clone https://github.com/NikoKaramitos/SenseRator2.0.git
    cd SenseRator2.0
  ``` 
4. **Prepare the YOLOv8 Model:**  
Place your custom-trained YOLOv8 model `(Final50Epochs.pt)` in the project directory.

## Usage  

1. **Running the Code:**
   
    Simply run the Python script to start detecting, tracking, and recording video:  
  
    ```
       python camera_detection.py
    ```
    The application will start the video stream, perform object detection and tracking, and calculate various safety indices based on detected objects.

2. **Video Recording:**  
    The video stream will be automatically recorded and saved to the videos/ directory with a unique filename based on the current timestamp `(e.g., output_YYYYMMDD_HHMMSS.mp4)`.
  
    Videos are saved in `.mp4` format using the `mp4v` codec, and the color is properly converted to RGB.  

3. **Pedestrian Flow Safety Index:**  
    After each run, the program will calculate and display the Pedestrian Flow Safety (PFS) index, along with individual indices for sidewalks, crosswalks, traffic lights, stop signs, trees, and street lights.

## Code Structure

- `frame_capture_thread():`  
  Captures frames from the camera in real-time and pushes them into a processing queue.
  
- `inference_thread():`  
  Processes each frame using YOLOv8 for object detection and ByteTrack for object tracking. The tracking age and object counts are updated here.
  
- `display_thread():`  
  Annotates the detected objects on the video frames, converts the frames to the correct RGB format, and writes them to the video file while displaying them in real-time.
  
- `compute_component_score():`  
  Calculates the score for each detected object category, contributing to the overall Pedestrian Flow Safety (PFS) index.

## Output

- **Console Output:** The program prints the object counts, detection confidence scores, tracking IDs, and the final calculated safety indices.
- **Video Output:** Annotated video streams with object bounding boxes and IDs are saved as `.mp4` files in the `videos/` directory.

## Example

  After running the program, the console output will display something like:
  
  ```
  Detected Crosswalk with confidence 0.82 and track_id 5
  Detected Sidewalk with confidence 0.78 and track_id 2
  ...
  Final Object Counts:
  Crosswalk: 3
  Sidewalk: 2
  Traffic Light: 1
  Stop Sign: 0
  Tree: 6
  Street Light: 2
  
  Pedestrian Flow Safety Index: 85.50
  =====================================
  Sidewalk Index:        100.00
  Crosswalk Index:       80.00
  Traffic Light Index:   50.00
  Stop Sign Index:       0.00
  Tree Index:            100.00
  Street Light Index:    75.00
  ```
  
  The annotated video will be saved in the `videos/` folder.
