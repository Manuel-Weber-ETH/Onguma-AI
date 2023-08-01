import os
import cv2
import numpy as np
import shutil
import sys
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
from tensorflow.keras.models import load_model
from tkinter.ttk import Progressbar
import warnings
import logging

# Set up logging
log_file = "onguma_ai_log.txt"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

# Suppress TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to suppress the TensorFlow warning
def suppress_tensorflow_warning(message, category, filename, lineno, file=None, line=None):
    pass

# Suppress the TensorFlow warning
warnings.showwarning = suppress_tensorflow_warning

# Default prediction probability threshold
prediction_threshold = 0.85

# Load the trained model
model_file_path = "C:/0_Documents/2_Projects/Namibia - Onguma/AI Software/APP/Rendering to .exe/rhino_detection_model.h5"
model = load_model(model_file_path)

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
    logging.debug(f"Predicting class for image: {image_path}")
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    logging.debug(f"Prediction probability: {prediction}")
    return prediction


def classify_images(folder_path):
    logging.debug("Classifying images")
    total_images = count_images_in_folder(folder_path)
    progress_bar['maximum'] = total_images

    rhino_folder = os.path.join(folder_path, "rhino")
    not_rhino_folder = os.path.join(folder_path, "not_rhino")

    os.makedirs(rhino_folder, exist_ok=True)
    os.makedirs(not_rhino_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_classified = 0

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        prediction_prob = predict_class(image_path)

        # Check against the user-defined threshold
        prediction_result = "Rhino" if prediction_prob >= prediction_threshold else "Other Animal"

        destination_folder = rhino_folder if prediction_result == "Rhino" else not_rhino_folder
        shutil.move(image_path, os.path.join(destination_folder, image_file))
        total_classified += 1
        remaining_images = total_images - total_classified
        progress_label.config(text=f"Progress: {total_classified}/{total_images} (Remaining: {remaining_images})")
        progress_bar['value'] = total_classified
        root.update_idletasks()

    messagebox.showinfo("Success", "Image classification completed successfully.")

def count_images_in_folder(image_folder_path):
    num_images = len([f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))])
    return num_images

def browse_folder():
    logging.debug("Browse button clicked")
    folder_path = filedialog.askdirectory(title="Select the folder containing images to classify")
    if folder_path:
        if os.path.exists(folder_path):
            num_images = count_images_in_folder(folder_path)
            messagebox.showinfo("Images Found", f"Number of images found: {num_images}")
            logging.debug(f"Number of images found: {num_images}")
            classify_images(folder_path)
        else:
            messagebox.showerror("Error", "Invalid folder path. Please select a valid folder.")
            logging.error("Invalid folder path selected")

# Create the GUI
root = tk.Tk()
root.title("Onguma AI")  # Set the window title to "Onguma AI"
root.configure(bg="white")  # Set the background color to white

# Define window width and height
window_width = 600
window_height = 400

# Load and display a custom logo image
logo_image_path = "C:/0_Documents/2_Projects/Namibia - Onguma/AI Software/APP/Rendering to .exe/Onguma-logo.jpg"
if os.path.exists(logo_image_path):
    logo_image = Image.open(logo_image_path)
    logo_image.thumbnail((100, 100))  # Keep the aspect ratio and resize to fit 100x100 pixels
    logo_photo = ImageTk.PhotoImage(logo_image)
else:
    # If the logo image is missing, create a placeholder image
    logo_image = Image.new("RGB", (100, 100), (255, 255, 255))
    logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(root, image=logo_photo)
logo_label.pack(padx=10, pady=(10, 0), anchor="se")  # Position the logo in the bottom-right corner

# Welcome message
welcome_label = tk.Label(root, text="Welcome to Onguma AI", font=("Perpetua", 16), bg="white", fg="black")
welcome_label.pack(pady=10)

# Load and display an image
image_path = "C:/0_Documents/2_Projects/Namibia - Onguma/AI Software/APP/Rendering to .exe/Onguma Black Rhino.jpg" # Replace with the path to your image file
if os.path.exists(image_path):
    image = Image.open(image_path)
    # Keep the aspect ratio while resizing to fit the label
    image.thumbnail((window_width, window_height // 2))
    photo = ImageTk.PhotoImage(image)
else:
    # If the default image is missing, create a blank image with a white background
    image = Image.new("RGB", (window_width, window_height // 2), (255, 255, 255))
    photo = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=photo)
image_label.pack()

# Function to set the classification threshold
def set_threshold():
    try:
        threshold = float(threshold_entry.get())
        if 0.0 <= threshold <= 1.0:
            global prediction_threshold
            prediction_threshold = threshold
        else:
            messagebox.showerror("Error", "Please enter a valid threshold between 0.0 and 1.0.")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid numeric threshold.")

# Label and Entry for Prediction Probability Threshold
threshold_label = tk.Label(root, text="Prediction Probability Threshold:")
threshold_label.pack(pady=5)

threshold_entry = tk.Entry(root, justify='center')
threshold_entry.insert(0, str(prediction_threshold))  # Set default threshold value
threshold_entry.pack(pady=5)

# Button to Set Prediction Probability Threshold
set_threshold_button = tk.Button(root, text="Set Threshold", command=set_threshold)
set_threshold_button.pack(pady=5)

browse_button = tk.Button(root, text="Click to select the folder containing images to classify", command=browse_folder)
browse_button.pack()

# Progress bar and label
progress_label = tk.Label(root, text="Progress: 0/0")
progress_label.pack(pady=5)

progress_bar = Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=5)

# Link to a website and some text
website_link = "https://www.onguma-ai.com"  # Replace with your desired website link
link_label = tk.Label(root, text="Visit our website: www.Onguma-AI.com", fg="black", cursor="hand2", font=("Perpetua", 12, "underline"))
link_label.pack(pady=5)

# Function to open the website link
def open_website(event):
    import webbrowser
    webbrowser.open(website_link)

link_label.bind("<Button-1>", open_website)

# Some additional text
additional_text = tk.Label(root, text="© Manuel Weber, Onguma Nature Reserve copyright 2023", font=("Perpetua", 6))
additional_text.pack(pady=5)

root.mainloop()
logging.debug("Onguma AI application closed")