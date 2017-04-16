# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Architecture"
[image2]: ./images/center_2017_04_15_09_16_04_256.jpg "Center Lane Image"
[image3]: ./images/center_2017_04_15_09_26_59_843.jpg "Recovery Right Image"
[image4]: ./images/center_2017_04_15_09_27_50_765.jpg "Recovery Left Image"
[image5]: ./images/center_2017_04_15_09_16_47_178.jpg "Original Turn Image"
[image6]: ./images/center_2017_04_15_09_16_47_178_f.jpg "Flipped Turn Image"
[image7]: ./images/training.png "Validation Loss Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4 containing video of autonomous driving recorded by video.py
* run2.mkv containing one more video with better quality recorded from OS to see all driving parameters

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 kernel sizes (model.py lines 80-84)

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 79).

See next section for detailed model architecture.

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting (model.py lines 88).

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, images from left and right cameras and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first idea was to use NVIDIA model from [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). As it was quite large I decided to try more simple model first and then upgrade it. But then I found that my simpler model is enough to drive the car on first track and so I submit it.

#### 2. Final Model Architecture

First, input data is cropped in the model to use road part of the image only (model.py line 76).

Then data is normalized in the model using a Keras lambda layer (line 79).

Then my model has three convolution layers with 5x5 kernels and 6, 16, 32 numbers of filters (lines 80, 82, 84).

Each convolution layer uses RELU as activation to introduce nonlinearity (lines 80, 82, 84).

Each convolution layer is followed by max pooling layer (lines 81, 83, 85).

Also the model has four fully connected layers with 100, 50, 10 and 1 elements (line 87, 89-91) and one dropout layer with 0.5 probability (line 88).

Here is a visualization of the architecture made automatically by Keras
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get the correct position if something went wrong. I recorded one lap of zigzag driving:
* drive right to the lane
* start recording
* drive to the center
* stop recording
* drive left
* start recording
* drive to the center
* stop recording

Here are some examples of recovering driving:

![alt text][image3]![alt text][image4]

To augment the data sat, I also flipped images to have the same number of right and left turns. For example, here is an image that has then been flipped:

![alt text][image5] ![alt text][image6]

Then I wanted to add images from left and right cameras with some correction to the steering angles. But after looking at images I decided to add them without any correction as they looked valid. Not to confuse the model by different images (center, left and right) for every steering angle I added images from left and right cameras with 0.25 probability to one center camera image. Also I didn't add left and right images for recovery driving.

After the collection process, I had 4822 center lane driving images from center camera, 3036 recovery driving images. All that images were flipped, so I got (4822 + 3036) * 2 = 15716 images. Also about 1200 left camera and 1200 right camera images.
So the result number of data points was about 18120.

 I then preprocessed this data by cropping 70 pixels from the top and 20 pixels from the bottom. Cropping was done right in the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The final number of epochs I used was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here are results of training the model using AWS GPU instance:
```
Epoch 1/10
175/174 [==============================] - 42s - loss: 0.0363 - val_loss: 0.0198
Epoch 2/10
175/174 [==============================] - 32s - loss: 0.0192 - val_loss: 0.0184
Epoch 3/10
175/174 [==============================] - 30s - loss: 0.0178 - val_loss: 0.0181
Epoch 4/10
175/174 [==============================] - 30s - loss: 0.0171 - val_loss: 0.0169
Epoch 5/10
175/174 [==============================] - 30s - loss: 0.0162 - val_loss: 0.0164
Epoch 6/10
175/174 [==============================] - 31s - loss: 0.0157 - val_loss: 0.0157
Epoch 7/10
175/174 [==============================] - 30s - loss: 0.0154 - val_loss: 0.0152
Epoch 8/10
175/174 [==============================] - 30s - loss: 0.0147 - val_loss: 0.0146
Epoch 9/10
175/174 [==============================] - 30s - loss: 0.0144 - val_loss: 0.0140
Epoch 10/10
175/174 [==============================] - 30s - loss: 0.0139 - val_loss: 0.0141
```
![alt text][image7]

So it took about 5 minutes to train the model.

### Conclusion
Actually, it wasn't hard project, so now I want to create a model and collect training data for the second track.

To complete the project I haven't to invent something new, I just used some of advices from the lectures for the project. I didn't preprocess images a lot and didn't experiment a lot with model architecture as in traffic sign project. First model with cropped images worked quite well.

The only trouble I had was collecting data. I tried to drive as smooth as possible to have more images with right steering angle instead of many images with zero angle and some images with large angles.

Also I had some problems with AWS because I didn't access to GPU instances first, then I got NVIDIA driver problem (I found solution on Udacity forum) and finally the model trained with python 3.5 couldn't be run on my computer with python 3.6 and I have to install one more python + tensorflow + keras + opencv on my desktop. It makes sense to solve such problems or just notice next students.

As you may have the same problems to run the model as I had I have recorded two videos of autonomous driving. I didn't recognize well the start point of the first track and so recorded till the bridge appeared second time (so video is a bit longer than one track driving). run1.mp4 if a video file generated by drive.py and video.py, it has poor quality and doesn't contain speed, angle and "Mode: Autonomous" text. So I recorded run2.mkv grabbing it from the simulator window.
