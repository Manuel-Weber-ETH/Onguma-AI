import os
import cv2
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:/0_Documents/2_Projects/Namibia - Onguma/AI Software/rhino_detection_model.h5')

# Define the input image size
input_size = (224, 224)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of the image
def predict_class(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    return prediction

# Function to move images to appropriate folders
def move_images_to_folders(image_folder_path):
    rhino_folder = os.path.join(image_folder_path, "rhino")
    not_rhino_folder = os.path.join(image_folder_path, "not_rhino")

    os.makedirs(rhino_folder, exist_ok=True)
    os.makedirs(not_rhino_folder, exist_ok=True)

    subfolders = [f for f in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, f))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_folder_path, subfolder)
        image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            prediction_prob = predict_class(image_path)
            prediction_result = "Rhino" if prediction_prob >= 0.8 else "Other Animal"

            print(f"Image: {image_file} | Probability of Rhino: {prediction_prob:.4f} | Classification: {prediction_result}")

            destination_folder = os.path.join(rhino_folder, subfolder) if prediction_result == "Rhino" else os.path.join(not_rhino_folder, subfolder)
            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(image_path, os.path.join(destination_folder, image_file))

    # Delete empty subfolders
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_folder_path, subfolder)
        if not os.listdir(subfolder_path):  # Check if the subfolder is empty
            os.rmdir(subfolder_path)

# Example usage
input_folder_path = 'C:/0_Documents/2_Projects/Namibia - Onguma/AI Software/Test folder'
move_images_to_folders(input_folder_path)
