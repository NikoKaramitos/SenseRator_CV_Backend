from tracemalloc import stop
from ultralytics import YOLO
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy
import cv2
import threading
import queue
from collections import defaultdict
import os
import datetime

# Load your custom-trained YOLOv8 model with ByteTrack
model = YOLO('Final50Epochs.pt')
print("Model Loaded")

# Initialize the camera and display
camera = videoSource("csi://0", argv=["--input-width=640", "--input-height=480", "--framerate=20"])  # Default CSI camera source
display = videoOutput("display://0")  # Display window

# Create a queue to hold frames for inference
frame_queue = queue.Queue(maxsize=4)  # Adjust maxsize as needed

# Variable to store the latest tracking results
latest_results = None
results_lock = threading.Lock()

# Initialize object count dictionary
object_counts = defaultdict(set)
object_counts_lock = threading.Lock()

# Initialize tracking ages
track_ages = {}
track_ages_lock = threading.Lock()

# Set minimum track age for counting objects
min_track_age = 2  # Adjust as needed

if not os.path.exists('videos'):
    os.makedirs('videos')

# Initialize feature scores
crosswalk_score = 0
sidewalk_score = 0
traffic_light_score = 0
street_light_score = 0
stop_sign_score = 0
tree_score = 0
pfs_score = 0

def compute_component_score(class_name, ids):
    num_objects = len(ids)
    score = 0
    global crosswalk_score
    global sidewalk_score
    global traffic_light_score
    global street_light_score
    global stop_sign_score
    global tree_score

    if class_name == 'sidewalk':
        if num_objects >= 1:
            score = 100
        
        sidewalk_score = score
        score *= 0.25

    if class_name == 'Crosswalk':
        if num_objects >= 3:
            score = 100
        elif num_objects == 2:
            score = 50
        elif num_objects == 1:
            score = 25
        
        crosswalk_score = score
        score *= 0.2
    
    if class_name == 'Traffic Light':
        if num_objects >= 2:
            score = 100
        elif num_objects == 1:
            score = 50
        
        traffic_light_score = score
        score *= 0.15

    if class_name == 'stop':
        if num_objects >= 2:
            score = 100
        elif num_objects == 1:
            score = 50
            
        stop_sign_score = score
        score *= 0.15

    if class_name == 'tree':
        if num_objects >= 6:
            score = 100
        elif num_objects in (4, 5):
            score = 50
        elif num_objects >= 1:
            score = 25
            
        tree_score = score
        score *= 0.1

    if class_name == 'Street_Light':
        if num_objects >= 3:
            score = 100
        elif num_objects == 2:
            score = 50
        elif num_objects == 1:
            score = 25
        
        street_light_score = score
        score *= 0.15
    
    return score

def inference_thread():
    global latest_results
    while True:
        # Get a frame from the queue
        np_img = frame_queue.get()
        if np_img is None:
            break  # Exit the thread if None is received

        # Resize image for faster inference
        resized_img = cv2.resize(np_img, (320, 256))

        # Run inference with ByteTrack for tracking
        results = model.track(
            resized_img,
            imgsz=(320, 256),
            conf=0.1,
            persist=True,
            tracker="bytetrack.yaml"  # Use ByteTrack for tracking
        )

        # Scale bounding boxes back to original image size
        scale_x = np_img.shape[1] / resized_img.shape[1]
        scale_y = np_img.shape[0] / resized_img.shape[0]

        # Prepare tracking results for display
        tracked_objects = []
        tracked_ids = set()  # To keep track of unique IDs seen this frame

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    if box.xyxy is None or len(box.xyxy) == 0:
                        continue  # Skip this detection

                    # Get bounding box coordinates and class info
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    conf = box.conf.item()
                    cls = int(box.cls.item())

                    if cls < 0 or cls >= len(model.names):
                        continue

                    class_name = model.names[cls]
                    track_id = int(box.id.item()) if box.id is not None else None

                    # Update the track age and object count
                    if track_id is not None:
                        with track_ages_lock:
                            if track_id in track_ages:
                                track_ages[track_id] += 1  # Increment age
                                print(f"Tracking ID: {track_id}, Age: {track_ages[track_id]}")
                            else:
                                track_ages[track_id] = 1  # Initialize age
                                print(f"New track detected: ID: {track_id}, Class: {class_name}")

                            # Only count the object if it meets the minimum age
                            if track_ages[track_id] >= min_track_age:
                                tracked_ids.add(track_id)
                                with object_counts_lock:
                                    object_counts[class_name].add(track_id)  # Add ID to counts
                                print(f"Counting ID: {track_id}, Class: {class_name}, Age: {track_ages[track_id]}")

                    tracked_objects.append((track_id, class_name, int(x1), int(y1), int(x2), int(y2), conf))

        # Update the latest results
        with results_lock:
            latest_results = tracked_objects

        # Indicate that the frame has been processed
        frame_queue.task_done()

# Start the inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# Generate a unique filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'videos/output_{timestamp}.mp4'

# Define the codec and create VideoWriter object for .mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

try:
    # Run the camera loop
    while display.IsStreaming():

        # Capture frame from the camera
        try:
            img = camera.Capture()
            if img is None:
                continue
        except Exception as e:
            print(f"Camera capture error: {e}")
            continue

        # Convert the captured image from CUDA to NumPy
        np_img = cudaToNumpy(img)

        # Try to add the frame to the queue for inference
        try:
            frame_queue.put_nowait(np_img.copy())
        except queue.Full:
            pass  # Skip adding the frame if the queue is full

        # Get the latest tracking results
        with results_lock:
            tracks = latest_results

        # If we have tracking results, overlay them on the frame
        if tracks is not None:
            for obj in tracks:
                track_id, class_name, x1, y1, x2, y2, conf = obj
                label = f'ID: {track_id} {class_name} {conf:.2f}'

                # Draw bounding box and label
                cv2.rectangle(np_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(np_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Write the frame to the video file
        video_writer.write(np_img_rgb)

        # Convert the NumPy image back to CUDA format before rendering
        cuda_img = cudaFromNumpy(np_img)

        # Render the CUDA image to the display
        display.Render(cuda_img)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Clean up
    frame_queue.put(None)  # Signal the inference thread to exit
    thread.join()
    camera.Close()
    display.Close()

    video_writer.release()

    # After exiting the loop, print the final counts
    print("\nFinal Object Counts:")
    with object_counts_lock:
        for class_name, ids in object_counts.items():
            print(f"{class_name}: {len(ids)}")  # Show count of unique IDs for each class

            # Calculate the score for the Pedestrian Flow Safety Index
            pfs_score += compute_component_score(class_name, ids)

    # Print all final calculations
    print(f"\nPedestrian Flow Safety Index: {pfs_score: .2f}")
    print("=====================================")
    print(f"Sidewalk Index:\t\t{sidewalk_score: .2f}")
    print(f"Crosswalk Index:\t{crosswalk_score: .2f}")
    print(f"Traffic Light Index:\t{traffic_light_score: .2f}")
    print(f"Stop Sign Index:\t{stop_sign_score: .2f}")
    print(f"Tree Index:\t\t{tree_score: .2f}")
    print(f"Street Light Index:\t{street_light_score: .2f}")