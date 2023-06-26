import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as no
import warnings

warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
  
import os
import matplotlib.image as mpimg


# Data Import
data_path = 'training_set'

classes = os.listdir(data_path)
#print(classes)


# Data Vizualization
fig = plt.gcf()
fig.set_size_inches(8, 8)

cats_dir = os.path.join('training_set/cats')
dogs_dir = os.path.join('training_set/dogs')

cats_names = os.listdir(cats_dir)
dogs_names = os.listdir(dogs_dir)

pic_index = 200

cats_images = [os.path.join(cats_dir, fname)
               for fname in cats_names[pic_index-8:pic_index]]
dogs_images = [os.path.join(dogs_dir, fname)
               for fname in dogs_names[pic_index-8:pic_index]]

for i, img_path in enumerate(cats_images + dogs_images):
    sp = plt.subplot(4, 4, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)
#plt.show()


# Training Preparation
train_data = image_dataset_from_directory(data_path, 
                                          image_size= (200, 200),
                                          subset= 'training',
                                          seed= 1,
                                          validation_split= 0.1,
                                          batch_size= 32)
test_data = image_dataset_from_directory('test_set',
                                         image_size= (200, 200),
                                         subset= 'validation',
                                         seed= 1,
                                         validation_split= 0.1,
                                         batch_size= 32)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])
#model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Model Training
model.fit(train_data, epochs=10, validation_data=test_data)
model.save('model_1')