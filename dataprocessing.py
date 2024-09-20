import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Set image dimensions and path to dataset
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CLASSES = 43  # Number of classes in GTSRB dataset

def load_data(data_dir):
    images = []
    labels = []
    
    # Read all classes and images
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Load data
data_dir = "path_to_gtsrb_dataset"  # Replace with your dataset path
images, labels = load_data(data_dir)

# Normalize images
images = images / 255.0

# One-hot encode labels
labels = to_categorical(labels, NUM_CLASSES)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)



#data augmentation 

from keras.preprocessing.image import ImageDataGenerator

# Augmentation strategy for training images
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

# Apply augmentation to training data
datagen.fit(X_train)



