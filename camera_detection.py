from ultralytics import YOLO
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy
import cv2
import threading
import queue
import time
from collections import defaultdict
import numpy as np
import os
import datetime


# Load your custom-trained YOLOv8 model on CPU
model = YOLO('Final50Epochs.pt')
print("Model Loaded on CPU")

# Initialize the camera and display
camera = videoSource("csi://0", argv=["--input-width=640", "--input-height=480", "--framerate=20"])
display = videoOutput("display://0")  

# Initialize object count dictionary
object_counts = defaultdict(set)
object_counts_lock = threading.Lock()

# Event to signal threads to exit
exit_event = threading.Event()

# Shared variables and locks
processed_frame_lock = threading.Lock()
latest_processed_frame = None

track_ages = {}
track_ages_lock = threading.Lock()
min_track_age = 3 

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

# Calculate individual scores and weights
def compute_component_score(class_name, ids):
    num_objects = len(ids)
    score = 0

    global crosswalk_score
    global sidewalk_score
    global traffic_light_score
    global street_light_score
    global stop_sign_score
    global tree_score

    if (class_name == 'sidewalk'):
        if (num_objects >= 1):
            score = 100

        sidewalk_score = score
        score *= 0.25

    if (class_name == 'Crosswalk'):
        if (num_objects >= 3):
            score = 100
        elif (num_objects == 2):
            score = 50
        elif (num_objects == 1):
            score = 25

        crosswalk_score = score
        score *= 0.2
    
    if (class_name == 'Traffic Light'):
        if (num_objects >= 2):
            score = 100
        elif (num_objects == 1):
            score = 50

        traffic_light_score = score
        score *= 0.15

    if (class_name == 'stop'):
        if (num_objects >= 2):
            score = 100
        elif (num_objects == 1):
            score = 50

        stop_sign_score = score
        score *= 0.15

    if (class_name == 'tree'):
        if (num_objects >= 6):
            score = 100
        elif (num_objects == 4 or 5):
            score = 50
        elif (num_objects >= 1):
            score = 25

        tree_score = score
        score *= 0.1

    if (class_name == 'Street_Light'):
        if (num_objects >= 3):
            score = 100
        elif (num_objects == 2):
            score = 50
        elif (num_objects == 1):
            score = 25

        street_light_score = score
        score *= 0.15
    
    return score


def frame_capture_thread():
    while not exit_event.is_set():
        try:
            # Capture frame from the camera
            img = camera.Capture()
            if img is None:
                continue

            # Convert the captured image from CUDA to NumPy
            np_img = cudaToNumpy(img)

            # Put the frame into the frame queue
            frame_queue.put(np_img)
        except Exception as e:
            print(f"Camera capture error: {e}")
            continue

def inference_thread():
    global latest_processed_frame
    while not exit_event.is_set():
        try:
            # Get a frame from the frame queue with timeout
            try:
                np_img = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Measure inference time
            start_time = time.time()

            # Reduce input resolution for faster inference
            resized_img = cv2.resize(np_img, (320, 256)) 

            # Run inference with built-in tracking
            results = model.track(
                resized_img,
                imgsz=(320, 256),
                conf=0.1,
                persist=True,
                tracker="bytetrack.yaml"
            )

            # Scale bounding boxes back to original image size
            scale_x = np_img.shape[1] / resized_img.shape[1]
            scale_y = np_img.shape[0] / resized_img.shape[0]

            # Prepare tracking results for display
            tracked_objects = []

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:

                        # Before processing box
                        if box.xyxy is None or len(box.xyxy) == 0:
                            print("Invalid bounding box data")
                            continue  # Skip this detection

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                        conf = box.conf.item()
                        cls = int(box.cls.item())

                        if cls < 0 or cls >= len(model.names):
                            print(f"Invalid class index: {cls}")
                            continue

                        class_name = model.names[cls]
                        track_id = int(box.id.item()) if box.id is not None else None

                        # Print confidence and tracking ID
                        print(f"Detected {class_name} with confidence {conf:.2f} and track_id {track_id}")

                        tracked_objects.append((track_id, class_name, int(x1), int(y1), int(x2), int(y2), conf))

                        # Update tracking age
                        if track_id is not None:
                            with track_ages_lock:
                                if track_id in track_ages:
                                    track_ages[track_id] += 1
                                else:
                                    track_ages[track_id] = 1

                            with object_counts_lock:
                                object_counts[class_name].add(track_id)

                            if track_ages[track_id] >= min_track_age:
                                print(f"Object counted: {class_name} with ID {track_id}")
                        else:
                            print(f"Assigned NONE for detection with confidence {conf:.2f}")
                else:
                    print("No boxes detected")

            # Measure inference time
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.2f} seconds")

            # Update the latest processed frame
            with processed_frame_lock:
                latest_processed_frame = (np_img.copy(), tracked_objects)

        except Exception as e:
            print(f"Inference error: {e}")
            continue

# Generate a unique filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'videos/output_{timestamp}.mp4'

# Define the codec and create VideoWriter object for .mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

def display_thread():
    while not exit_event.is_set():
        try:
            # Get the latest processed frame
            with processed_frame_lock:
                if latest_processed_frame is not None:
                    np_img, tracked_objects = latest_processed_frame
                    np_img = np_img.copy()  # Make a copy to prevent threading issues
                else:
                    continue  # No frame to display yet

            # Draw the tracks on the frame
            for obj in tracked_objects:
                track_id, class_name, x1, y1, x2, y2, conf = obj

                label = f'ID: {track_id} {class_name} {conf:.2f}'

                # Draw bounding box
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

            # Check if the display is still open
            if not display.IsStreaming():
                exit_event.set()
                break

            time.sleep(0.01)  # reduce CPU usage

        except Exception as e:
            print(f"Display error: {e}")
            continue

# Create the frame queue
frame_queue = queue.Queue(maxsize=30)

# Start the threads
capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
inference_thread_instance = threading.Thread(target=inference_thread, daemon=True)
display_thread_instance = threading.Thread(target=display_thread, daemon=True)

capture_thread.start()
inference_thread_instance.start()
display_thread_instance.start()

try:
    # Keep the main thread alive while other threads are running
    while display.IsStreaming():
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrupted by user")
    exit_event.set()
finally:
    # Signal the threads to exit
    exit_event.set()

    # Wait for threads to finish
    capture_thread.join(timeout=5)
    inference_thread_instance.join(timeout=5)
    display_thread_instance.join(timeout=5)

    # close the video writer
    video_writer.release()

    # Clean up resources
    camera.Close()
    display.Close()

    # After exiting the loop, print the final counts
    print("\nFinal Object Counts:")
    with object_counts_lock:
        print(f"Complete object_counts: {object_counts}")
        for class_name, ids in object_counts.items():
            print(f"{class_name}: {len(ids)}")

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
    
