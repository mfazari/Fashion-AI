Fashion AI
------------


Training a convolutional network using Keras and Tensorflow to recognize clothing pieces. Inspired by the [following project.](https://github.com/jasminevasandani/know-your-ai)

Getting Started
------------

***Disclaimer:*** *Fashion AI is a work in progress. The dataset will be expanded to provide better results. Functionality such as recognition of objects as well as further optimization and debug are planned to be added in the future.*

**Following libraries are being used:**
* keras
* tensorflow (version 1.15)
* numpy
* pandas
* matplotlib.pygot
* seaborn
* pillow

**Our dataset:**
1. Due to copyright reasons, our dataset is not included in this repository.
2. We use images provided by different fast fashion brand to train our model.
3. Has the following properties
    * Caps (62 images)
    * Pants (54 images)
    * Shoes (100 images)
    * Sweater (180 images)
    * T-Shirt (75 images)
4. Since this is a rather small sample to train our network, we use Keras ImageDataGenerator to augment our dataset in "fashion_create_model.py".
    


**The 3 main files:**
1. files_to_CSV.py
    * This file pre-processes our images. It takes our dataset and resizes all files to 75x110 pixels while also converting them to grayscale. Lastly, it uses numpy to save all data into an array that we convert into a .csv file afterwards.
2. fashion_create_model.py
    * Reads our .csv file and trains our neuronal network. We save the model as "model_fashion_2.h5" using Tensorflow.
3. fashion_test_model.py
    * Loading our created model into this file to test it. Output is found in /images/predictions.

**Output**
- Successful Prediction

    ![Alt text](images/predictions/prediction_82.png "Output")

Sources
------------
- [Keras Image PreProcessing](https://keras.io/preprocessing/image/)
- [Keras: Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)


