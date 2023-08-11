import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Set the path to your image directory
image_directory = 'directory'

# Define the input image size
input_size = (224, 224)

# Define batch size and number of training steps per epoch
batch_size = 32
steps_per_epoch = 7800

# Data augmentation to prevent overfitting
data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = data_generator.flow_from_directory(
    image_directory,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    image_directory,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load the MobileNetV2 model without the top classification layer
base_model = MobileNetV2(input_shape=(input_size[0], input_size[1], 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers to prevent overfitting
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
for epoch in range(5):  # Increase the number of epochs
    print(f"Epoch {epoch+1}/{5}")
    for step in range(steps_per_epoch):
        try:
            data_batch, labels_batch = next(train_generator)
            loss, accuracy = model.train_on_batch(data_batch, labels_batch)
            print(f"Step {step+1}/{steps_per_epoch} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Skipped batch due to error: {e}")
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

# Save the trained model
model.save('directory')

print("Training completed. Model saved as 'rhino_detection_model_250k_8_8.h5'.")
