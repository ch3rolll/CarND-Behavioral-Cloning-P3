# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used the deep learning approach which I have learned in the past weeks to train the simulator drive same just as I do.
For the best of my understanding of this project, the most important part is the training data, since the network is just copying and cloning what you did. It would help a lot to use some image processing to get a better image input for the network.

Data Collection
---
Based on the thoughts and velocity is not an input, as the drive.py uses a PI controller to have a constant speed. I just drove as slow as possible to get a better center position of the car in the image.

The dataset includes 3 laps of counter-clockwise and 2 laps of clockwise.

To avoid the fusion, I did not include recovery part, like how to pull back the vehicle from offtrack.

So it takes a lot of time to train the network for a given big amount of dataset.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies

The following resources are directly from Udacity Repo:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

Firstly, I followd the instruction of the class to use a simple one-layer conv2D, just for trying the workflow or the pipeline to actually train a network and sucessfully be saved. The result was pretty bad as expected. The network seemed not working at all.  But at least, it proved the pipeline was complete.

Then I switched to LeNet-5, since it works fine for the last project. But it was sort of slow to train and the reslut was not that good as expected. The car drove out of the track before the bridge.

Then I extended the dataset by using multi-cameras, but made a mistake that -correction to left image as I thought the left turnning angles are negtive.

Afterwards, I switched to Nvidia Self-driving car arch. It got a good result.
#### Reduce overfitting
3 dropout layers following fully connected network are used to reduce overfitting. Dropout rate = 0.5
#### Parameter tuning
Dropout rate: 1 -> 0.5
Learning rate: Adam is used for getting an adaptive learning rate
Epoch: with increasing the number of epochs, the result is getting better from 1 epoch to 3 epochs

### 2. Final Model Architecture

The final model architecture consisted of 6 conv2d layers, followed by 5 fully-connected network.

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving at a slow speed to make sure it stays at the center.

Then I repeated this process on the opposite direction in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the car to stay in the center. 

## Details About Files In This Directory

### model_nvi.py
Network: nvidia architecture
Dataset: images from left, right and center cameras + Flipped image and negative steering angles.
Number of epochs: 3

### model_nvi_e1.h5
Model was trained with 1 epoch since the dataset is quite big and it takes 5000s to finish one epoch, even using 2 GPUs

### model_nvi_e3.h5
Model was trained with 3 epochs

### run1.mp4
The video is recorded with using model_nvi_e1.h5. We can clearly see the car was stock on the bridge. So with more epochs involved, the result is getting better.

### epoch3.mp4
The video is recorded with using 3 epochs on same architecture. It managed to cross the bridge. But it drove out the track afterwards. We will see if 5 epochs help to get a better result.



