# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_histogram.png "Training Data"
[image2]: ./examples/valid_histogram.png "Validation Data"
[image3]: ./examples/test_histogram.png "Test Data"
[image4]: ./examples/internet1.jpg "Traffic Sign 1"
[image5]: ./examples/internet2.jpg "Traffic Sign 2"
[image6]: ./examples/internet3.jpg "Traffic Sign 3"
[image7]: ./examples/internet4.jpg "Traffic Sign 4"
[image8]: ./examples/internet5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amrosebirani/carnd-traffic-classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classed

Training data

![alt text][image1]

Validation data

![alt text][image2]

Test data

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize images.

For normalization I used the following code

```
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    y = np.empty(shape=[0,32,32,3])
    for d in x:
        min_val = np.min(d)
        max_val = np.max(d)
        y = np.append(y, [(d-min_val)/(max_val-min_val)], axis=0)
    print ('ok')
    return y
```

I tried adding grayscale but it didn't improve the accuracy as much so I decided to stay with the color images.

With this I was able to achieve greater then 93% accuracy on validation set in only 20 epochs.

However given all the details around improvement to accuracy after augmenting the dataset, I tried doind it, but generating the data is a long process, so I am still working on it [here] (https://github.com/amrosebirani/carnd-traffic-classification/blob/master/Traffic_Sign_Classifier_augmented.ipynb), but considering the current timeframe want to submit this and improve more later.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5X5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5X5     	| 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 				|
| Flatten	    | outputs 600      									|
| Fully connected		| outputs 150       									|
| RELU					|												|
| Dropout					|	.75									|
| Fully connected		| outputs 100       									|
| RELU					|												|
| Dropout					|	.75									|
| Fully connected		| outputs 43       									|
| Softmax				| etc.        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer
The loss operation was calculated using reduced mean on softmax cross entropy.

The training paramas are

```
EPOCHS = 20
BATCH_SIZE = 256
NB_CLASSES = 43
keep_probability = .75
rate = 0.001
```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.5%
* test set accuracy of 91.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
  
  I tried with the original lenet architecture, using grayscale and the aforementioned normalization, then moved onto the current architecture by first adding a convolution layer, and after that adding a dropout on the fully connected layers.

* What were some problems with the initial architecture?
  
  The initial architecture was underfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?  
  
  I added more complexity on the convolution layers by adding more filter layers, hence solving underfitting. Also to solve this further to remove overfitting I added dropout layers at the fully connected level. This improved the test accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
  
  Since we are performing image classification, convolution is a good choice to gather the low level features. Adding complexity at the convolution level helps find more features and then adding droput helps reduce overfitting.

If a well known architecture was chosen:
* What architecture was chosen?  
  
  Lenet

* Why did you believe it would be relevant to the traffic sign application?  
  
  It was originally used for image classification, hence it was an automatic choice.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
  
  With a very basic model without even generating more data I am able to get an accuracy of greater then 93%. With further improvements I can easily achieve greater then 98% accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth image might be difficult to classify because it has a lot of watermarks and it is not as clear as the others.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| Slippery road	      		| Slippery road					 				|
| Wild animals crossing			| Double curve      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 91.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999         			| Road Work   									| 
| 1.000     				| Right-of-way at the next intersection 										|
| 0.597					| Speed limit (30km/h)											|
| 0.974	      			| Slippery road					 				|
| 0.562				    | Double curve      							|


