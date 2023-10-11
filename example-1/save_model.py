#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Step 1: Load the pretrained model
model = ResNet50(weights="imagenet")

# Step 2: Save the model in SavedModel format
saved_model_path = "saved_model_directory"
tf.saved_model.save(model, saved_model_path)
