# SenseRator2.0: 
## YOLOv8 Object Detection and Tracking Project

This project utilizes a custom-trained YOLOv8 model for real-time object detection and tracking with video stream recording. The primary application is detecting various street-level objects (e.g., sidewalks, crosswalks, traffic lights, stop signs, etc.) and calculating safety-related indices to determine a **Pedestrian Flow Safety (PFS) index** based on detected objects.

### Features
- **Object Detection and Tracking:**
  Detects and tracks multiple objects using a YOLOv8 model with ByteTrack for object tracking.
- **Real-time Inference:**
   Processes video streams in real-time using the `jetson_utils` library for efficient handling of Jetson devices.
- **Score Calculation:**
   Computes safety-related indices based on the detected objects, providing insights into pedestrian safety.
- **Video Recording:**
   Records the video stream with object annotations and saves it to the `videos/` directory in `.mp4` format without overwriting previous recordings.devices.
- **Score Calculation:**
   Computes safety-related indices based on the detected objects, providing insights into pedestrian safety.
- **Video Recording:**
   Records the video stream with object annotations and saves it to the `videos/` directory in `.mp4` format without overwriting previous recordings.

### Requirements
- **Hardware:**
  - Jetson Nano or compatible device with a CSI camera.
- **Software:**
  - Python 3.8+
  - JetPack 4.6 or newer
  - `ultralytics` library for YOLOv8
  - `jetson_utils` for video capture and display
  - `opencv-python`
  - `numpy`

### Installation

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

### Usage  

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

### Code Structure

- `camera = videoSource("csi://0", ...)` & `display = videoOutput("display://0"):`    
  Initializes the camera feed from the CSI camera and sets up the display window for real-time video rendering.
  
- `frame_queue = queue.Queue(maxsize=4):`  
  A queue is created to hold frames before sending them for inference. The `maxsize` ensures that frames are processed without overloading the system, and new frames are dropped if the queue is full.
  
- `inference_thread():`  
  Runs object detection and tracking on each frame taken from `frame_queue`. It resizes the frame for faster inference and processes the results using ByteTrack to maintain object tracking across frames. Tracked objects are stored, including their bounding boxes, class labels, and confidence scores. The track ages are managed to ensure that only sufficiently tracked objects are counted.
  
- `thread = threading.Thread(target=inference_thread, daemon=True):`  
  Starts the inference thread in the background to handle detection and tracking concurrently with video rendering.
  
- `compute_component_score():`  
  Calculates a score for each detected object category (e.g., sidewalks, crosswalks, traffic lights) based on the number of objects detected. The scores contribute to the overall **Pedestrian Flow Safety (PFS) index.**
  
- `video_writer = cv2.VideoWriter(...):`  
  Initializes the video writer to record the video stream to an `.mp4` file in the `videos/` directory. The filename includes a timestamp to prevent overwriting.
  
- `camera.Capture()` & `cudaToNumpy()`  
  Captures frames from the camera feed and converts them from CUDA format to NumPy for further processing and rendering.
  
- **Main Loop** (`while display.IsStreaming()`):  
  - Captures frames from the camera feed
  - Adds frames to inference queue (if available space)
  - Renders tracking results onto the frames (bounding boxes, labels, confidence scores).
  - Writes frames to video file in `.mp4` format
  - Dispalys frames in real-time display window
 
- **Exception Handling and Cleanup** (`try`/`finally`):
  Gracefully handles interruptions (e.g., KeyboardInterrupt) and cleans up resources like the camera, display, and inference thread. Ensures the video writer is released and prints the final object counts along with the **PFS index** calculation.  

### Output

- **Console Output:** The program prints the object counts, detection confidence scores, tracking IDs, and the final calculated safety indices.
- **Video Output:** Annotated video streams with object bounding boxes and IDs are saved as `.mp4` files in the `videos/` directory.

### Example

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



## Dataset Augmentation and YOLOv8 Training

This script is the dataset augmentation pipeline and custom training setup for YOLOv8 models.   The script combines multiple datasets, remaps the labels, applies   augmentations, and prepares the data for YOLOv8 training.

### Features
- **Dataset Augmentation:**  
  Applies a series of augmentations including horizontal flips, brightness/contrast adjustments, rotations, and blurs to increase the dataset size and variability.
- **Label Remapping:**  
  Automatically remaps the class indices of different datasets to a unified set of class labels for training YOLOv8.
- **Data Combination and Splitting:**  
  Combines datasets of different categories (e.g., crosswalks, traffic lights, stop signs, trees, etc.) and splits them into training and validation sets.
- **Custom YOLOv8 Training:**  
  Trains a YOLOv8 model on the combined and augmented datasets using the `ultralytics` YOLO framework.

### Requirements
- **Software:**  
  - Python 3.8+
  - `opencv-python`
  - `albumentations`
  - `scikit-learn`
  - `ultralytics` (for YOLOv8)
  - `numpy`
  - `PyYAML`

### Installation

1. **Install Dependencies:**  
   First, make sure you have the necessary Python dependencies:
   ```
   pip install opencv-python albumentations scikit-learn ultralytics numpy pyyaml
   ```
2. **Prepare Datasets:**  
   Place your datasets in the correct paths as specified in the code. Each dataset should have `images/` and `labels/` directories inside their respective paths.

### Usage
#### Augmentation and Dataset Preparation  
1. **Dataset Augmentation:** The following script will augment the selected images from each dataset by applying random transformations such as flips, brightness adjustments, and rotations. Example of augmenting and combining datasets:
  ```
  for dataset_info in datasets_info:
    images, labels = select_and_augment_images(dataset_info, 1000, augment_factor=2)
  ```
2. **Remap Labels:** Each dataset might have a different label mapping. This code will remap the labels to a unified set of class indices based on the `class_mapping`.
  ```
  remap_labels(label_path, original_mapping)
  ```
3. **Combine and Split Datasets:** After augmentation, the dataset is combined and split into 80% training and 20% validation sets:
  ```
  train_images, val_images, train_labels, val_labels = train_test_split(combined_image_paths,  combined_label_paths, test_size=0.2, random_state=42)
  ```
4. **Save and Organize Combined Dataset:** The images and labels are copied to the `combined_dataset/` directory, with separate subdirectories for `train` and `val` data.
  ```
  copy_files(train_images, os.path.join(combined_images_dir, 'train'))
copy_files(val_images, os.path.join(combined_images_dir, 'val'))
  ```

### Custom YOLOv8 Training
1. **Train YOLOv8 Model:** train a YOLOv8 model on the prepared and combined dataset:
  ```
  model = YOLO('yolov8s.pt')
results = model.train(data=data_yaml_path, epochs=10, imgsz=640)
  ```
2. **Configure Dataset for Training:** The dataset configuration is stored in a YAML file that specifies the class names, training, and validation paths:
  ```
 data_config = {
    'nc': 8,
    'names': ['Crosswalk', 'Traffic Light', 'regulatory', 'stop', 'warning', 'tree', 'sidewalk', 'Street_Light'],
    'train': '<PATH TO YOUR TRAINING DATA>',
    'val': '<PATH TO YOUR TESTING DATA>'
}
  ```
3. **Save Configuration File:** The configuration is saved to a `data.yaml` file, which is passed to the YOLOv8 training function:
  ```
with open('<FILE PATH TO SAVE TRAINING FUNCTION>/data.yaml', 'w') as yaml_file:
    yaml.dump(data_config, yaml_file, default_flow_style=False)
  ```

### Example  
  After running the script, the program will augment and combine datasets, then train YOLOv8 on the augmented data. Example console output:
  ```
  Number of images in <FILE PATH TO TO YOUR SIDEWALK CROSSWALK DATASET>/train: 1000
  Number of images in <FILE PATH TO TO YOUR SIDEWALK TRAFFIC LIGHT DATASET>/train: 1000
  Number of images in <FILE PATH TO TO YOUR SIDEWALK TRAINING DATASET>/train: 1000
  ...
  Total number of images in the combined dataset: 6000
  Datasets combined, labels remapped, augmented, and split successfully!
  data.yaml file created successfully!
  ```
  Training results will be displayed during the training process, including loss values and performance metrics.

### Output
- **Augmented Images:**
  Augmented images are saved in the same directory as the original images with `_aug_N` appended to the filenames.
- **Combined Dataset:**
  The combined dataset is saved in the `combined_dataset/` directory, with separate directories for `train` and `val` images and labels.
- **YOLOv8 Training Results:**
  The YOLOv8 training results, including model weights and metrics, will be stored in the `runs/train` directory (default behavior of `ultralytics` library).
