import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pandas as pd
from PIL import Image 
from PIL.ImageDraw import Draw

width = 226
height = 226
num_classes = 2
classes = ["nrfp", "No-ntfp"]

TRAINING_CSV_FILE = 'Data/training_data.csv'
TRAINING_IMAGE_DIR = 'Images/training'

training_image_records = pd.read_csv(TRAINING_CSV_FILE)

train_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

train_images = []
train_targets = []
train_labels = []

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


for index, row in training_image_records.iterrows():
    
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row
    
    train_image_fullpath = os.path.join(train_image_path, filename)
    train_img = keras.preprocessing.image.load_img(train_image_fullpath, target_size=(height, width))
    # img_width, img_height = train_img.size

    # # Print the dimensions
    # print("Image Dimensions (width, height):", img_width, img_height)
    train_img_arr = keras.preprocessing.image.img_to_array(train_img)
    resized_img = train_img.resize((width, height))

    # Convert the resized image to a NumPy array
    resized_img_arr = keras.preprocessing.image.img_to_array(resized_img)
    # Print the dimensions of the resized image
    print("Resized Image Dimensions (width, height):", resized_img_arr.shape)
    
    
    xmin = round(xmin/ width, 2)
    ymin = round(ymin/ height, 2)
    xmax = round(xmax/ width, 2)
    ymax = round(ymax/ height, 2)
    
    train_images.append(resized_img_arr)
    train_targets.append((xmin, ymin, xmax, ymax))
    train_labels.append(classes.index(class_name))

for img_arr in train_images:
    print(img_arr.shape)

# for img_arr in train_images:
#     print(type(img_arr))


train_images = np.array(train_images)
train_targets = np.array(train_targets)
train_labels = np.array(train_labels)

# validation dataset#

TRAINING_CSV_FILE = 'Data/validation_data.csv'
TRAINING_IMAGE_DIR = 'Images/validation'

validation_image_records = pd.read_csv(TRAINING_CSV_FILE)

val_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

validation_images = []
validation_targets = []
validation_labels = []

for index, row in validation_image_records.iterrows():
    
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row
    
    val_image_fullpath = os.path.join(val_image_path, filename)
    val_img = keras.preprocessing.image.load_img(val_image_fullpath, target_size=(height, width))
    val_img_arr = keras.preprocessing.image.img_to_array(val_img)
    
    
    xmin = round(xmin/ width, 2)
    ymin = round(ymin/ height, 2)
    xmax = round(xmax/ width, 2)
    ymax = round(ymax/ height, 2)
    
    validation_images.append(val_img_arr)
    validation_targets.append((xmin, ymin, xmax, ymax))
    validation_labels.append(classes.index(class_name))

validation_images = np.array(validation_images)
validation_targets = np.array(validation_targets)
validation_labels = np.array(validation_labels)
    
#create the common input layer
input_shape = (height, width, 3)
input_layer = tf.keras.layers.Input(input_shape)

#create the base layers
base_layers = layers.experimental.preprocessing.Rescaling(1./255, name='bl_1')(input_layer)
base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
base_layers = layers.Flatten(name='bl_8')(base_layers)

#create the classifier branch
classifier_branch = layers.Dense(128, activation='relu', name='cl_1')(base_layers)
classifier_branch = layers.Dense(num_classes, name='cl_head')(classifier_branch)  

#create the localiser branch
locator_branch = layers.Dense(128, activation='relu', name='bb_1')(base_layers)
locator_branch = layers.Dense(64, activation='relu', name='bb_2')(locator_branch)
locator_branch = layers.Dense(32, activation='relu', name='bb_3')(locator_branch)
locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)

model = tf.keras.Model(input_layer,
           outputs=[classifier_branch,locator_branch])

losses = {"cl_head":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),"bb_head":tf.keras.losses.MSE}

model.compile(loss=losses, optimizer='Adam', metrics=['accuracy'])

trainTargets = {
    "cl_head": train_labels,
    "bb_head": train_targets
}
validationTargets = {
    "cl_head": validation_labels,
    "bb_head": validation_targets
}
training_epochs = 20

print("Train Images Shape:", np.array(train_images).shape)
print("Train Targets Shape:", np.array(trainTargets).shape)

history = model.fit(train_images, trainTargets,
             validation_data=(validation_images, validationTargets),
             batch_size=4,
             epochs=training_epochs,
             shuffle=True,
             verbose=1)