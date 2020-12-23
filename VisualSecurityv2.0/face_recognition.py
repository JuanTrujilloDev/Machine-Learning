# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:29:49 2020

@author: sarsu
"""



from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Re-sizing all the images to 224x244
IMAGE_SIZE = [224, 224]

#Taking the path of the images
train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

# Adding preprocessing layer to the front of VGG using 3 chanels and imagenet wieghts.
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Trainning only unexisting weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # Getting the # of classes
folders = glob('Datasets/Train/*')
  

# Defining activation function
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Creating the model output
model = Model(inputs=vgg.input, outputs=prediction)

# Displaying the model structure
model.summary()

# Telling the model what loss and optimization algorithms to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



# fitting the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs = 10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# Getting the loss Values
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Getting the accuracy values
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_accuracy')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')