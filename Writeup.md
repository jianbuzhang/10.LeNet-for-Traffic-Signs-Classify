# Traffic Sign Recognition

* Author:Jin Jian
* Time  2018/04/06

[//]: #(ImageLink)
[image1]:./examples/grayscale.jpg "Grayscaling"

---
## The main steps to accomplish this project**
The steps of this projcet are the following:
* Import related library and testing data
* Resize the images
* Include exploratory visualization
* Preprocess the data set(normalization,grayscale,etc.)
* Design a Convolutional Neural Network
* Train,Validate and Test the Model
* Predict the sign type for each image
* Analyze Performance
* Give out top five probablities for each image from the Internet

---
## Data Set Summary &Exploration

### 1.using python,numpy and/or pandas to provide a basic summary of the data set.

I used the pandas library to calculate the statistics,for example:
* the size of the training data set.
* the size of the validation .
* the shape of a traffic sign image.
* the number of unique classes or labels in the data set.

### 2.Pre-process the data set
* normalization.Here we add batch normalization(shorten as BN) method.For example, when neural networks are trained with slow convergence rates, or gradient explosions, which cannot be trained, you can try BN to solve them.Besides, BN can raising the training speed and strengthen the model accuracy.
* grayscale : Here is an example of the image comparison after gray processing.
![alt text][image1]
### 3.Include exploratory visualization
* By using matplotlib.pyplot library, I can randomly choose a image from the data set ,recognize and show it.
---
## Design and test a model architecture

### 1.The CNN structure is as following:
INPUT->CONV->RELU->POOL->CONV->RELU->POOL->FLAT->FC->RELU->FC->OUTPUT

The following dataframe shows the strcuture:

|Layer                  |   Description         |
|:---------------------:|:---------------------:|
|Input                  |32x32x3 RGB image      |
|Convolution 3x3        |1x1 stride,same_panding|
|RELU                   |                       |
|Max pooling            |2x2 stride,out 16x16x64|
|convolution 3x3        |etc.                   |
|FC                     |etc.                   |
|softmax                |etc.                   |
|output                 |n_classes              |

### 2.Goal: the final validation accuracy should be 0.88 or higher.
A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

In the conv2d function, use_cudnn_on_gpu=True, thus fasten the calculation rate.

By adjusting the parameters of EPOCHS and BATCH_SIZE,we can get different accuracy and different training time.The following dataframe recorded my debugging process.

|Count  |EPOCHS |BATCH_SIZE |validation accuracy|time       |
|:-----:|:-----:|:---------:|:-----------------:|:---------:|
|1      |10     |256        |0.873              |4min 2s    |
|2      |10     |128        |0.885              |4min       |
|3      |10     |96         |0.912              |4min 5s    |
|4      |14     |96         |0.894              |5min 39s   |
|5      |10     |64         |0.896              |4min 21s   |

During the debugging process,I found that the accuracy may fall down,like the below(EPOCHS=14, BATCH_SIZE=96):
* Training...
* EPOCH 1 ...
* Validation Accuracy = 0.706
* EPOCH 2 ...
* Validation Accuracy = 0.820
* EPOCH 3 ...
* Validation Accuracy = 0.845
* EPOCH 4 ...
* Validation Accuracy = 0.834
* EPOCH 5 ...
* Validation Accuracy = 0.860
* EPOCH 6 ...
* Validation Accuracy = 0.853
* EPOCH 7 ...
* Validation Accuracy = 0.880
* EPOCH 8 ...
* Validation Accuracy = 0.883
* EPOCH 9 ...
* Validation Accuracy = 0.891
* EPOCH 10 ...
* Validation Accuracy = 0.888
* EPOCH 11 ...
* Validation Accuracy = 0.900
* EPOCH 12 ...
* Validation Accuracy = 0.888
* EPOCH 13 ...
* Validation Accuracy = 0.896
* EPOCH 14 ...
* Validation Accuracy = 0.894
* Model saved
* Wall time: 5min 39s

Compared to the debugging datas, finally I choose the third figures(EPOCHS=10,BATCH_SIZE=96) and the accuracy is 0.911,test time is 4min.

Besides, according to the data, when we set the EPOCH too large, sometimes the accuracy would drop down instead of getting better.

### 3.Predict the sign type for given images
In my code, I used softmax function to identify the probablities for each prediction when predicting on each of the five new images.Besides , I have also listed the top five softmax probablities for the images.

* For the first image:

|   Count   | Probability               |   Prediction(%)   | 
|:---------:|:-------------------------:|:-----------------:|
|1          |Speed limit(70km/h)        |99.93              |
|2          |Speed limit(30km/h)        |0.07               |
|3          |Speed limit(120km/h)        |0                 |
|4          |Keep left                  |0                  |
|5          |Speed limit(50km/h)        |0                  |

* For second image:

|   Count   | Probability               |   Prediction(%)   | 
|:---------:|:-------------------------:|:-----------------:|
|1          |Road work                  |100                |
|2          |Bicycle crossing           |0                  |
|3          |Beware of ice/snow         |0                  |
|4          |Slippery road              |0                  |
|5          |Wild animals crossing      |0                  |

* For third image:

|   Count   | Probability               |   Prediction(%)   | 
|:---------:|:-------------------------:|:-----------------:|
|1          |Stop                       |100                |
|2          |Yield                      |0                  |
|3          |Turn right ahead           |0                  |
|4          |Ahead only                 |0                  |
|5          |Turn left ahead            |0                  |

* For fourth image:

|   Count   | Probability               |   Prediction(%)   | 
|:---------:|:-------------------------:|:-----------------:|
|1          |Turn left ahead            |100                |
|2          |Ahead only                 |0                  |
|3          |Speed limit(60km/h)        |0                  |
|4          |Keep right                 |0                  |
|5          |Speed limit(30km/h)        |0                  |

* For fifth image:

|   Count   | Probability               |   Prediction(%)   | 
|:---------:|:-------------------------:|:-----------------:|
|1          |Speed limit(60km/h)        |99.99              |
|2          |Speed limit(80km/h)        |0.01               |
|3          |Ahead only                 |0                  |
|4          |Speed limit(50km/h)        |0                  |
|5          |Speed limit(100km/h)       |0                  |


