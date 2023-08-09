import os
import cv2
import numpy as np
import shutil
import sys
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageFont
from tensorflow.keras.models import load_model
from tkinter.ttk import Progressbar
import warnings
import logging
import threading
from collections import namedtuple
Font = namedtuple("Font", ["family", "size"])

# Define your custom font
custom_font = Font(family="Calibri", size=12)

# Explicitly set TensorFlow backend session to avoid potential issues
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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
prediction_threshold = 0.997

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load the trained model
model_file_path = resource_path("rhino_detection_model_250k_8_8.h5")
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

# Function to predict the class using a loaded model
def predict_class_with_model(model, image_path):
    logging.debug(f"Predicting class for image: {image_path}")
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    logging.debug(f"Prediction probability: {prediction}")
    return prediction

# Function to count the number of image files in a folder
def count_images_in_folder(image_folder_path):
    num_images = len([f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))])
    return num_images

# Function to generate a new name for a rhino photo
def generate_rhino_photo_name(waterhole, week, year, unique_number):
    new_name = f"R_{waterhole}_{year}_{week}_{unique_number}.jpg"
    return new_name

# Function to classify images and rename rhino photos
def classify_and_rename_images(file_path, waterhole, week, year):
    logging.debug("Classifying images")
    total_images = count_images_in_folder(file_path)
    algorithm_selected = algorithm_var.get()  # Get the selected algorithm

    # Load the appropriate model based on the selected algorithm
    if algorithm_selected == "Algorithm L (250k)":
        model_file = "rhino_detection_model_250k_8_8.h5"
    elif algorithm_selected == "Algorithm S (15k)":
        model_file = "rhino_detection_model.h5"
    else:
        model_file = None  # Default value
    
    if model_file:
        model_path = resource_path(model_file)
        model = load_model(model_path)
    else:
        messagebox.showerror("Error", "Invalid algorithm selection")
        return

    rhino_folder = os.path.join(file_path, f"R_{waterhole}_{year}_W{week}")
    not_rhino_folder = os.path.join(file_path, f"Other_{waterhole}_{year}_W{week}")

    os.makedirs(rhino_folder, exist_ok=True)
    os.makedirs(not_rhino_folder, exist_ok=True)

    image_files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]

    unique_number = 1

    def classify_thread(total_classified, unique_number):
        for image_file in image_files:
            image_path = os.path.join(file_path, image_file)
            
            # Use the loaded model for prediction
            prediction_prob = predict_class_with_model(model, image_path)

            # Check against the user-defined threshold
            prediction_result = "Rhino" if prediction_prob >= prediction_threshold else "Other_Animal"

            destination_folder = rhino_folder if prediction_result == "Rhino" else not_rhino_folder

            if prediction_result == "Rhino":
                new_name = generate_rhino_photo_name(waterhole, week, year, unique_number)
                unique_number += 1
                new_path = os.path.join(destination_folder, new_name)
                shutil.move(image_path, new_path)
            else:
                new_path = os.path.join(destination_folder, image_file)
                shutil.move(image_path, new_path)

            total_classified += 1

            # Update progress bar and label
            progress_value = int((total_classified / total_images) * 100)
            progress_bar["value"] = progress_value
            progress_label.config(text=f"Progress: {total_classified}/{total_images}")
            root.update_idletasks()  # Update the GUI immediately

            remaining_images = total_images - total_classified
            print(f"Classified: {image_file}, Probability: {prediction_prob:.4f}")

        messagebox.showinfo("Success", "Image classification and renaming completed successfully.")

    # Create a thread for classification
    classify_thread_instance = threading.Thread(target=classify_thread, args=(0, unique_number))
    classify_thread_instance.start()



# Set up dark mode colors
dark_bg_color = "#1f1f1f"
light_text_color = "white"

# Create the GUI
root = tk.Tk()
root.title("Onguma AI")
root.configure(bg=dark_bg_color)  # Set dark mode background color

# Load and display an image
image_path = resource_path("Onguma Black Rhino.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    # Keep the aspect ratio while resizing to fit the label
    image.thumbnail((400, 300))
    photo = ImageTk.PhotoImage(image)
else:
    # If the default image is missing, create a blank image with a dark background
    image = Image.new("RGB", (400, 300), (31, 31, 31))
    photo = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=photo, bg=dark_bg_color)
image_label.grid(row=0, column=0, rowspan=9, padx=20, pady=10, sticky="w")

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

# Dropdown menu for algorithm selection
algorithm_label = tk.Label(root, text="Select Algorithm:", font=custom_font, bg=dark_bg_color, fg=light_text_color)
algorithm_label.grid(row=6, column=1, padx=20, pady=10, sticky="e")

algorithm_var = tk.StringVar()
algorithm_var.set("Algorithm L (250k)")  # Set default algorithm selection
algorithm_options = ["Algorithm L (250k), p = 0.997", "Algorithm S (15k), p = 0.85"]

algorithm_menu = tk.OptionMenu(root, algorithm_var, *algorithm_options)
algorithm_menu.grid(row=6, column=2, padx=20, pady=10, sticky="w")

# Label and Entry for Prediction Probability Threshold
threshold_label = tk.Label(root, text="Prediction Probability Threshold:", font=custom_font, bg=dark_bg_color, fg=light_text_color)
threshold_label.grid(row=7, column=1, padx=20, pady=10, sticky="e")  # Use grid instead of pack

threshold_entry = tk.Entry(root, justify='center', font=custom_font)
threshold_entry.insert(0, str(prediction_threshold))  # Set default threshold value
threshold_entry.grid(row=7, column=2, padx=20, pady=10, sticky="w")  # Use grid instead of pack

# Button to Set Prediction Probability Threshold
set_threshold_button = tk.Button(root, text="Set Threshold", command=set_threshold)
set_threshold_button.grid(row=8, column=1, columnspan=2, padx=20, pady=10, sticky="e")  # Use grid instead of pack


# Function to browse and select a folder
def browse_folder():
    global file_path
    logging.debug("Browse button clicked")
    file_path = filedialog.askdirectory(title="Select the folder containing images to classify")
    if file_path:
        if os.path.exists(file_path):
            num_images = count_images_in_folder(file_path)
            messagebox.showinfo("Images Found", f"Number of images found: {num_images}")
            logging.debug(f"Number of images found: {num_images}")
        else:
            messagebox.showerror("Error", "Invalid folder path. Please select a valid folder.")
            logging.error("Invalid folder path selected")

# Entry fields for customizable elements
waterhole_label = tk.Label(root, text="Waterhole Name (B#):", font=custom_font, bg=dark_bg_color, fg=light_text_color)
waterhole_label.grid(row=0, column=1, padx=20, pady=10, sticky="e")

waterhole_entry = tk.Entry(root, justify='center', font=custom_font)
waterhole_entry.grid(row=0, column=2, padx=20, pady=10, sticky="w")

week_label = tk.Label(root, text="Week Number (W##):", font=custom_font, bg=dark_bg_color, fg=light_text_color)
week_label.grid(row=1, column=1, padx=20, pady=10, sticky="e")

week_entry = tk.Entry(root, justify='center', font=custom_font)
week_entry.grid(row=1, column=2, padx=20, pady=10, sticky="w")

year_label = tk.Label(root, text="Year (####):", font=custom_font, bg=dark_bg_color, fg=light_text_color)
year_label.grid(row=2, column=1, padx=20, pady=10, sticky="e")

year_entry = tk.Entry(root, justify='center', font=custom_font)
year_entry.grid(row=2, column=2, padx=20, pady=10, sticky="w")

# Button to browse and select a folder
browse_button = tk.Button(root, text="Select Folder", command=browse_folder, font=custom_font, bg=dark_bg_color, fg=light_text_color)
browse_button.grid(row=3, column=1, columnspan=2, padx=20, pady=10, sticky="e")

# Button to classify images and rename rhino photos
classify_button = tk.Button(root, text="Classify and Rename Rhino Photos", command=lambda: classify_and_rename_images(file_path, waterhole_entry.get(), week_entry.get(), year_entry.get()), font=(custom_font.family, custom_font.size), bg=dark_bg_color, fg=light_text_color)
classify_button.grid(row=10, column=1, columnspan=2, padx=20, pady=10, sticky="e")

# Progress bar and label
progress_label = tk.Label(root, text="Progress: 0/0", font=custom_font, bg=dark_bg_color, fg=light_text_color)
progress_label.grid(row=9, column=1, columnspan=2, padx=20, pady=10, sticky="w")

progress_bar = Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.grid(row=9, column=1, columnspan=2, padx=20, pady=10, sticky="e")

# Link to a website and some text
website_link = "https://www.onguma-ai.com"  # Replace with your desired website link
link_label = tk.Label(root, text="Visit our website: www.Onguma-AI.com", fg=light_text_color, cursor="hand2", font=("Calibri", 12, "underline"), bg=dark_bg_color)
link_label.grid(row=9, column=0, columnspan=2, padx=20, pady=10, sticky="w")

# Function to open the website link
def open_website(event):
    import webbrowser
    webbrowser.open(website_link)

link_label.bind("<Button-1>", open_website)

# Some additional text
additional_text = tk.Label(root, text="© Manuel Weber 2023 - Version 1.5 - Developed for Onguma Nature Reserve to the specifications of Jonathan Strijbis", font=("Calibri", 6), bg=dark_bg_color, fg=light_text_color)
additional_text.grid(row=10, column=0, columnspan=2, padx=20, pady=10, sticky="w")

root.mainloop()
logging.debug("Onguma AI application closed")
