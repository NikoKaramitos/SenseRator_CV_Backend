import albumentations as A
import cv2
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import numpy as np

# Augmentation pipeline
def augment_image(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.Blur(blur_limit=3, p=0.1),
    ])
    augmented = transform(image=image)

    if not isinstance(augmented['image'], np.ndarray):
        raise TypeError(f"Augmented image is not an ndarray, got: {type(augmented['image'])}")

    return augmented['image']

# Class mapping based on combined data.yaml
class_mapping = {
    'Crosswalk': 0,
    'traffic light': 1,
    'regulatory': 2,
    'stop': 3,
    'warning': 4,
    'tree': 5,
    'sidewalk': 6,
    'Street_Light': 7
}

# Function to remap the label indices
def remap_labels(label_path, original_mapping):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    remapped_lines = []
    for line in lines:
        components = line.strip().split()
        original_class_idx = int(components[0])
        new_class_idx = class_mapping[original_mapping[original_class_idx]]
        components[0] = str(new_class_idx)
        remapped_lines.append(' '.join(components) + '\n')

    with open(label_path, 'w') as file:
        file.writelines(remapped_lines)

# Sample original mappings for each dataset
crosswalks_original_mapping = {0: 'Crosswalk'}
traffic_lights_original_mapping = {0: 'traffic light'}
us_road_signs_original_mapping = {0: 'regulatory', 1: 'stop', 2: 'warning'}
trees_original_mapping = {0: 'tree'}
sidewalk_original_mapping = {0: 'sidewalk'}
street_light_original_mapping = {0: 'Street_Light'}

# List of datasets and their mappings
datasets_info = [
    {'path': '/home/al921245/Documents/Training/crosswalks-2/train', 'mapping': crosswalks_original_mapping},
    {'path': '/home/al921245/Documents/Training/traffic-light-detection-1/train', 'mapping': traffic_lights_original_mapping},
    {'path': '/home/al921245/Documents/Training/US-Road-Signs-72/train', 'mapping': us_road_signs_original_mapping},
    {'path': '/home/al921245/Documents/Training/Tree-Detection-1/train', 'mapping': trees_original_mapping},
    {'path': '/home/al921245/Documents/Training/sidewalk-1/train', 'mapping': sidewalk_original_mapping},
    {'path': '/home/al921245/Documents/Training/Lighting-3/train', 'mapping': street_light_original_mapping}
]

# Function to select and augment images
def select_and_augment_images(dataset_info, num_images, augment_factor=2):
    dataset_path = dataset_info['path']
    original_mapping = dataset_info['mapping']

    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')

    # Get all images in the dataset
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.JPG')]

    # Randomly select the desired number of images
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    selected_image_paths = [os.path.join(image_dir, img) for img in selected_images]
    selected_label_paths = [os.path.join(label_dir, img.replace('.jpg', '.txt')) for img in selected_images]

    augmented_images = []
    augmented_labels = []

    # Remap the labels for original images and apply augmentation
    for image_path, label_path in zip(selected_image_paths, selected_label_paths):
        # Read and augment the image
        image = cv2.imread(image_path)
        remap_labels(label_path, original_mapping)

        augmented_images.append(image_path)
        augmented_labels.append(label_path)

        # Apply augmentations
        for i in range(augment_factor - 1):  # We already have the original image, so only augment the rest
            augmented_image = augment_image(image)

            # Save augmented image and copy corresponding label
            augmented_image_path = image_path.replace('.jpg', f'_aug_{i}.jpg')
            cv2.imwrite(augmented_image_path, augmented_image)

            augmented_label_path = label_path.replace('.txt', f'_aug_{i}.txt')
            shutil.copy(label_path, augmented_label_path)

            augmented_images.append(augmented_image_path)
            augmented_labels.append(augmented_label_path)

    return augmented_images, augmented_labels

# Collect images and labels from each dataset
combined_image_paths = []
combined_label_paths = []

for dataset_info in datasets_info:
    images, labels = select_and_augment_images(dataset_info, 1000, augment_factor=2)
    print(f"Number of images in {dataset_info['path']}: {len(images)}")
    combined_image_paths.extend(images)
    combined_label_paths.extend(labels)

print(f"Total number of images in the combined dataset: {len(combined_image_paths)}")

# Split data into train and validation (80% train, 20% validation)
train_images, val_images, train_labels, val_labels = train_test_split(
    combined_image_paths, combined_label_paths, test_size=0.2, random_state=42
)

# Function to copy files to the combined dataset directory
def copy_files(file_paths, dest_dir):
    for file_path in file_paths:
        shutil.copy(file_path, os.path.join(dest_dir, os.path.basename(file_path)))

# Copy images and labels to the corresponding directories
combined_images_dir = 'combined_dataset/images'
combined_labels_dir = 'combined_dataset/labels'
os.makedirs(os.path.join(combined_images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(combined_images_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(combined_labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(combined_labels_dir, 'val'), exist_ok=True)

copy_files(train_images, os.path.join(combined_images_dir, 'train'))
copy_files(val_images, os.path.join(combined_images_dir, 'val'))
copy_files(train_labels, os.path.join(combined_labels_dir, 'train'))
copy_files(val_labels, os.path.join(combined_labels_dir, 'val'))

print("Datasets combined, labels remapped, augmented, and split successfully!")

data_config = {
    'nc': 8,
    'names': ['Crosswalk', 'traffic light', 'regulatory', 'stop', 'warning', 'tree', 'sidewalk', 'Street_Light'],
    'train': '/home/al921245/Documents/Training/combined_dataset/images/train',
    'val': '/home/al921245/Documents/Training/combined_dataset/images/val'
}

# Save the config to a YAML file
with open('/home/al921245/Documents/Training/combined_dataset/data.yaml', 'w') as yaml_file:
    yaml.dump(data_config, yaml_file, default_flow_style=False)

print("data.yaml file created successfully!")

data_yaml_path = '/home/al921245/Documents/Training/combined_dataset/data.yaml'

model = YOLO('yolov8n.pt')
results = model.train(data = data_yaml_path, epochs = 50, imgsz = (480, 320))