Classification of Handwritten digits (MNIST) using one hidden layer Neural Network in Numpy (Python 3)
---

©EsterHlav, 27 July 2017


The goal of the project is to build, train and test a simple neural network using only Numpy as a computational library. The difficulty mainly comes from the calculation of gradient (of objective function regarding the parameters) and finding the appropriate backpropagation equations.

## Architecture of Neural Network
![One hidden layer Neural Network with size $[3,4,2]$](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/400px-Artificial_neural_network.svg.png)

We use an architecture with only one hidden layer of 500 units. We use as activation function **tanh** and as output **softmax**. We also use the **cross-entropy loss** since this is a multi-classification problem.

---

## Dataset

MNIST ([official page](http://yann.lecun.com/exdb/mnist/))
Set of **28 x 28** pixels images: training data of 60,000 images and test data of 10,000 images. The labels are in the set *{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}*, making the prediction a **10-class** classification problem.

##### Example of digits:
![MNIST digits](https://www.researchgate.net/profile/Yantao_Wei2/publication/264900175/figure/fig6/AS:295900862795776@1447559671013/Fig-6-Examples-of-the-MNIST-dataset.png "mnist examples")

*Note:* Before being fed to the Neural Network, each image is normalized, *e.g.* the image is substracted from its mean and divised by its standard deviation.

---

## Utilisation

### Requirements
Run the following command in unix terminal to install the packages needed to run along Python 3:
```bash
$ pip3 install numpy matplotlib Pillow
```
### Training of NN
Run the following command in unix terminal:
```bash
$ python3 train.py
```
Modifications of parameters available in **train.py**.
By default the Neural Network is trained with the following parameters:
- Number of hidden units: **500**
- Learning rate: **0.005**
- Decay: **0.0002**
- Regularization: **0.015**
- Number of epochs: **150**

Also, the input layer has size 28 x 28 = 784 and output layer is of size 10.

The equations of the Neural Network are:

![first equations](https://github.com/EsterHlav/Neural-Network-Digits-Classifier-MNIST-Database/raw/master/eq1.png)

From this, using the Chain's rule we can find the gradient equations for all parameters (used for Gradient Descent):

![second equations](https://github.com/EsterHlav/Neural-Network-Digits-Classifier-MNIST-Database/raw/master/eq2.png)


### Test with 20 random test images
Run the following command in unix terminal:
```bash
$ python3 main.py
```

The test can be done with the 3 different trained models: **test50epch**, **test100epch** and **test150epch**. Their error rate on the test set are respectively: 50.98%, 14.57% and 13.61%

#### Demo
![gif animation](https://github.com/EsterHlav/Neural-Network-Digits-Classifier-MNIST-Database/raw/master/mainNN.gif "main test")

### Test with GUI to draw digit and predict class
Run the following command in unix terminal:
```bash
$ python3 guiNN.py
```
As the Neural Network was trained on the MNIST dataset, we need to preprocess the drawn image as similar as possible to MNIST. Therefore we use a binary threshold, a inner-limit-box of 20 x 20 pixels and a linear wrapping for centering the digits.

#### Demo
![gif animation](https://github.com/EsterHlav/Neural-Network-Digits-Classifier-MNIST-Database/raw/master/guiNN.gif "gui demo")

---

## Organization of code

### NN•py
Define the class NeuralNetwork. All the computations are made in Numpy arrays and matrices. Backpropagation equations are hard-coded for tanh and softmax using cross-entropy loss. Use **support•py** for most of annex functions, use also **Numpy**.

### train•py
Build a Neural Network and train it according to parameters provided, and then save it in pickle format to reuse later. Also, option to graph the evolution of loss, training error and validation error. Depends on **NN•py** and **support•py**.

### main•py
Test the Neural Network provided on 20 random images from the test set. Depends on **NN•py** and **support•py**.

### guiNN•py
Open GUI using tkinter (for GUI) to draw a digit on a squared canvas. Image is then processed and predicted using PIL (image wrapper), Scipy (image transformation) and Numpy (matrix wrapper). Depends on **MouseDrawing•py**, **PIL**, **Scipy** and **Numpy**.

### MouseDrawing•py
Define canvas and function to draw with tkinter (for GUI). Depends on **tkinter**.

---

## Conclusion of project

I discovered how much a simple one layer Neural Network tends to overfit as much as possible on MNIST. Indeed, after having trained the network, I realized that the good results on the test sets were due to the high degree of similarity in all the digits of MNIST. However, when drawing digits on a canvas, the classification was not as accurate. Only the help of a similar preprocess of the drawing close to the one on MNIST helped improving the predictions. A Convolutional Neural Network would have certainly be less sensitive to different images as it captures visual patterns, explaining clearly how much more accurate is a CNN.

---

##### References:
[1. Good explanation of Backpropagation equations and their derivation ](http://neuralnetworksanddeeplearning.com/chap2.html)

[2. Alternative explanation of Backpropagation equations](http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/)

[3. Source of inspiration for Numpy implementation of Neural Nerwork](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)

[4. Great introduction to cross-entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/#cross-entropy)

[5. How to preprocess new drawings like MNIST](https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4)
