import os
import csv
import pandas
import cv2
import sklearn
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Activation, Lambda, Conv2D, pooling, Cropping2D, Dropout

# Set GPU environment

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Data preparing

samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

correction =0.2
dropout = 0.5


# Just train images from center camera

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            images_flipped = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images_flipped.append(np.fliplr(center_image))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train = np.append(X_train, images_flipped,axis=0)
            y_train = np.append(y_train, np.negative(y_train),axis=0)
            yield sklearn.utils.shuffle(X_train, y_train)


# Generate data from all cameras

def generator_all(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            images_flipped = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images_flipped.append(np.fliplr(center_image))
                
                # For the left images
                left_name = './IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)
                images_flipped.append(np.fliplr(left_image))
                
                # For the right images
                right_name = './IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)
                images_flipped.append(np.fliplr(right_image))
                
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train = np.append(X_train, images_flipped,axis=0)
            y_train = np.append(y_train, np.negative(y_train),axis=0)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator_all(train_samples, batch_size=32)
validation_generator = generator_all(validation_samples, batch_size=32)


# Model build-up. Using Nvidia Arch.
model = Sequential()

# normalization
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))

model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
    
model.compile(loss='mse', optimizer='adam')


# Using 1 epoch to try the model out, using 3 epochs to get a better result

model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3)

model.save('model_nvi_e3.h5')
