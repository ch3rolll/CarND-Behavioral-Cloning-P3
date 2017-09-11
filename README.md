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
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The following resources are directly from :
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### model_nvi.py
Network: nvidia architecture
Dataset: images from left, right and center cameras + Flipped image and negative steering angles.
Number of epochs: 3

### model_nvi_ep1.h5: saved model trained with 1 epoch since the dataset is quite big and it takes 5000s to finish one epoch, even using 2 GPUs




