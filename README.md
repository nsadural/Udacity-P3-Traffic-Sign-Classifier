# **Traffic Sign Recognition** 
## Nikko Sadural

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/Visualizations/visualization_train.jpg "Training Data Visualization"
[image2]: ./data/Visualizations/visualization_valid.jpg "Validation Data Visualization"
[image3]: ./data/Visualizations/visualization_test.jpg "Test Data Visualization"
[image4]: ./data/webdata/children_crossing.jpg "Children crossing sign from web"
[image5]: ./data/webdata/slippery_road.jpg "Slippery road sign from web"
[image6]: ./data/webdata/speed_limit_30.jpg "Speed limit 30 km/h sign from web"
[image7]: ./data/webdata/stop.jpg "Stop sign from web"
[image8]: ./data/webdata/turn_right_ahead.jpg "Turn right ahead sign from web"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The following bar charts display the counts for the individual class IDs in each of the training, validation, and test datasets, respectively. It is interesting to see how the datasets have relatively high counts for labels like 1, 2, 12, 13, and 38, while other labels have significantly lower counts. This may have an effect on the predicted labels of other data outside of these datasets.

![alt text][image1]

![alt text][image2]

![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale using the skimage library because I did not want color to have as much of a significant effect on the prediction for the sign label as the shapes and characters of the signs.

As a last step, I normalized the image data because the processed data should have zero mean with small magnitudes such that calculations do not involve large numbers.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image 			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| flattened input 400, outputs 120				|
| RELU with dropout		| probability of keeping 0.50 for training		|
| Fully connected		| input 120, outputs 84         				|
| RELU with dropout		| probability of keeping 0.50 for training		|
| Fully connected		| input 84, outputs 43          				|
|:---------------------:|:---------------------------------------------:| 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model using the stochastic gradient descent Adam optimizer. My batch size was 128, and my number of epochs was 30. My learning rate was 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995.
* validation set accuracy of 0.967.
* test set accuracy of 0.938.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture that was chosen was the LeNet neural network architecture due to its proven capability from previous exercises.

* What were some problems with the initial architecture?
The problem with the initial architecture was that there is no regularization techniques to reduce overfitting with a larger dataset.

* How was the architecture adjusted and why was it adjusted?
Dropout regularization was implemented between the three fully connected layers of the network with a keep probability of 0.5 during training. This was implemented to reduce overfitting effects on the results of the validation dataset.

* Which parameters were tuned? How were they adjusted and why?
The number of epochs was increased from 20 to 30 to achieve a higher validation accuracy in exchange for a few more iterations. The output layer of the network was changed from 10 to 43 to account for the 43 labels in the dataset.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A convolution layer works well with this problem because CNNs are able to learn from correlated features of various representations of traffic sign images. Another advantage is weight-sharing, such that feature location or orientation has negligible effect on the model prediction, which is useful when a traffic sign's location could vary in the field of view of an image or video. The two dropout layers boosted the validation accuracy over 0.93 by reducing the model's dependence on the fully available dataset during training, thus regularizing and reducing overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

The first image may be difficult to classify due to its similarity in shape and color with the second image.
The third image may be difficult to classify due to another portion of a traffic sign being visible in the image.
The fourth image may be difficult to classify since it is highly distorted.
The fifth image may be difficult to classify because of its ovular shape instead of circular and similarity to other signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing		| Beware of Ice/Snow							| 
| Slippery Road			| Slippery Road									|
| Speed Limit 30 km/h	| Speed Limit 30 km/h							|
| Stop  	      		| Ahead Only					 				|
| Turn Right Ahead		| Speed Limit 30 km/h  							|


The model was only able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is not very good compared to the accuracy of 93.8% on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Beware of Ice/Snow sign (probability of 0.866), but the image contains a Children Crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .866         			| Beware of Ice/Snow							| 
| .113     				| Right-of-way at the next intersection			|
| .020					| Children crossing								|
| .000039      			| Slippery Road					 				|
| .000037			    | Road narrows on the right						|


For the second image, the model is relatively sure that this is a Slippery Road sign (probability of 0.887), and the image does contain a Slippery Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .887         			| Slippery Road     							| 
| .113     				| Wild animals crossing             			|
| .00009				| Dangerous curve to the left					|
| .000006     			| Double curve					 				|
| .000000002  		    | Right-of-way at the next intersection			|


For the third image, the model is relatively sure that this is a Speed Limit 30 km/h sign (probability of 1.0), and the image does contain a Speed Limit 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed Limit 30 km/h  							| 
| .00000000002			| Speed Limit 50 km/h               			|
| .00000000000006		| Speed Limit 70 km/h       					|
| .000000000000000004	| Speed Limit 20 km/h			 				|
| .00000000000000000000002 | Speed Limit 80 km/h                		|


For the fourth image, the model is relatively sure that this is an Ahead Only sign (probability of 0.967), but the image contains a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .967         			| Ahead Only        							| 
| .029     				| Speed Limit 60 km/h               			|
| .002  				| No passing                					|
| .001      			| Turn Left Ahead				 				|
| .0005     		    | Children Crossing                 			|


For the fifth image, the model is relatively sure that this is a Speed Limit 30 km/h sign (probability of 0.58), but the image contains a Turn Right Ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58         			| Speed Limit 30 km/h  							| 
| .29     				| Turn Right Ahead                     			|
| .09   				| Roundabout Mandatory      					|
| .01       			| Speed Limit 100 km/h			 				|
| .009      		    | Right-of-way at the next intersection			|

Overall, the prediction accuracy of the web image data can be improved by adding augmented data to the training dataset. There is a significantly higher count of some class IDs than others, which leads to biased predictions of images outside of the training dataset. The validation and test datasets performed very well since their data are distributed similarly to the training dataset.