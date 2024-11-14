from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil
import yaml
import random
from sklearn.model_selection import train_test_split

rf = Roboflow(api_key="YOURAPIKEY")

project = rf.workspace("projects-xa3tf").project("senserator-2.0-crosswalks")
version = project.version(3)
dataset = version.download("yolov8")

project = rf.workspace("projects-xa3tf").project("senserator-2.0-benches")
version = project.version(2)
dataset = version.download("yolov8")

project = rf.workspace("us-road-signs-projects").project("us-road-signs")
version = project.version(72)
dataset = version.download("yolov8")

project = rf.workspace("innovasur").project("tree-detection-x5ic3")
version = project.version(1)
dataset = version.download("yolov8")

project = rf.workspace("projects-5k1o6").project("sidewalk-dlu6l")
version = project.version(1)
dataset = version.download("yolov8")

project = rf.workspace("projects-xa3tf").project("senserator-2.0-street_lights")
version = project.version(2)
dataset = version.download("yolov8")