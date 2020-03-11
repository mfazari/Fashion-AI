# Basic libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
# import os

# Modeling libraries
from sklearn.model_selection import train_test_split

# ConvNet Processing Libraries
# from PIL import Image
import keras
from keras.utils import np_utils, to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

# Model evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# For reproducibility
np.random.seed(2019)





# Check keras version
import keras

keras.__version__

# # Import image data


# Read in image data
image_data = pd.read_csv('./data/image_pixels.csv')

# Show shape of data
print(f'Shape: {image_data.shape}')
print(f'There are {image_data.shape[0]} images in this dataset!')

# Drop column in DataFrame
image_data.drop(columns=['Unnamed: 0'], inplace=True)

# Preview head
image_data.head(2)

# ### Create predictor `X` variable

# My `X` variable will contain image pixels. I need to pre-process my `X` variable to make sure it's ready to be run through my model. The only pre-processing step I need to take here is reshaping the array. I explain more a few cells below.


# Create variable to hold only pixel data
pixels = [col for col in image_data.columns if (col != 'piece_num_labels') & (col != 'pieces_string_labels')]

# Preview image pixel values
image_data[pixels].head(2)

# Notice in the DataFrame above that the image pixels are still flattened, meaning that you can't tell what the
# dimensions of the images are. What I'm going to do next, in the cell below, is reshape these pixels so that they
# resemble my image dimensions. Instead of each row corresponding to one image, I need one image to have 110 rows and
# 75 columns--and an array is the format to do this in. The 110 rows corresponds to the images' height, and the 75
# columns corresponds to the images' width. The values of the pixels are a range between 0-255. I talk about this in
# more detail later.


# Convert DataFrame to array, and reshape to a three-dimensional array
X = image_data[pixels].values.reshape(image_data.shape[0], 110, 75)

# Check shape of X array
print(f'The shape of the X array is: {X.shape}')

# Preview what `X` looks like. This is the first image in the dataset, with pixel range 0-255.
X[0]

# ### Create target `y` variable

# My `y` variable will contain pieces labels (the pre-assigned piece for each image).


# Convert labels column to array
y = image_data['piece_num_labels'].values

# Check shape of y array
print(f'The shape of the y array is: {y.shape}')

print(y[0:10])
print(y[1026:])

# ## Visualize image pixels to make sure everything was processed correctly


# Plot images
fig, axes = plt.subplots(3, 3, figsize=(6, 8))
ax = axes.ravel()
plt.suptitle('Fashion database', y=1.03, size=16)

ax[0].imshow(X[40], cmap=plt.cm.gray)
ax[0].set_title("caps")
ax[1].imshow(X[100], cmap=plt.cm.gray)
ax[1].set_title("pants")
ax[2].imshow(X[200], cmap=plt.cm.gray)
ax[2].set_title("shoes")
ax[3].imshow(X[250], cmap=plt.cm.gray)
ax[3].set_title("sweater")
ax[4].imshow(X[400], cmap=plt.cm.gray)
ax[4].set_title("t_shirt")

fig.tight_layout()
plt.show()

# # Train/Test Split

# Train/test split is taking my original dataset of image pixels and image labels, and splitting it into a training set and test set. I will "train" my model on my training set, and then I'll use my testing set as if it's unseen data and pass it through my model. Train/test split will help me determine if my model will perform well on more unseen data.
#
# After train/test splitting, I need to conduct additional pre-processing to my data to make sure it's ready to be run through my CNN models. In summary, I conducted the following pre-processing steps in this section:
#  - 1. Convert pixel values (0-255) to percentages (e.g., 1.0 to indicate 100%, 0.75 to indicate 75%, etc)
#  - 2. Reshape `X` array from `(110, 75)` t `(110, 75, 1)`
#  - 3. Convert `y` piece interger values 0-4 to categorical values


# Train/test split, and stratify y since this is a classification problem
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y)  # Stratify for classification

# An array of arrays.
# Each array is an individual image, each value is the value range of a pixel (0-255)
X_train[0]

# Each image is 110 pixels by 75 pixels
print(f'Image dimensions: {X_train[0].shape}')

# Save a copy of these arrays before processing
# I can use these unaltered arrays to make predictions later
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

# ### Convert `X_train`/`X_test` values to percentages

# In the cell below, I'm converting my pixel values to floats and then dividing them by 255 because I want to get a percentage value between 0 and 1. Now if a pixel is 1 (or 100%), that means it's white. And if it's 0 (or 0%) that means it's black. If the pixel is 0.75, then that means the pixel is probably going to be a light gray, and if it's 0.25 then it's going to be a dark gray. Converting my pixel values to percentages between 0 and 1 helps increase processing time by increasing computational efficiency.

# In[18]:


# Convert each value to a float, since we want to divide and get a percentage value.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# This scales each value to be between 0 and 1 (0 - 100%)
X_train /= 255
X_test /= 255

# Check scaled values
X_train[0][0]

# Check shape
print(f'Shape of training set: {X_train.shape}')

# Check shape
print(f'Shape of testing set: {X_test.shape}')

# ### Reshape `X_train`/`X_test` arrays

# Reshape images
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Check new reshape (making it show only one since X_train[0][0] shows a long list of arrays)
X_train[0][0][0]

# New shape of training set
print(f'New shape of training data: {X_train.shape}')
print(f'There are {X_train.shape[0]} images in my train dataset.')
print('-' * 40)
# New shape of training set
print(f'New shape of testing data: {X_test.shape}')
print(f'There are {X_test.shape[0]} images in my test dataset.')

# Show shape of y train
print(f'Shape of y_train: {y_train.shape}')

# Show shape of y test
print(f'Shape of y_test: {y_test.shape}')

# Preview of y labels
y_train[0:5]

# ### Convert `y_train`/`y_test` to categorical values

# Save a non-converted y_test for when I make predictions. I'll need it.
y_test_copy = y_test.copy()

# Convert y to a categorical variable, 0-1 (0-100%)
y_train = np_utils.to_categorical(y_train, 5)
y_test = np_utils.to_categorical(y_test, 5)

# Check to see that changes were made, a preview of the first three image labels
y_train[0:3]

# Show shape of y train
print(f'Shape of reshaped y_train: {y_train.shape}')

# Show shape of y test
print(f'Shape of reshaped y_test: {y_test.shape}')

# # Baseline Score


# Show total number of images per piece in entire dataset
image_data['pieces_string_labels'].value_counts(normalize=True)

# Show class with the highest percentage
image_data['pieces_string_labels'].value_counts(normalize=True)[[0]]

# Print statement to say what the baseline score is
baseline_score = round(image_data['pieces_string_labels'].value_counts(normalize=True)[0], 3) * 100
print(f'My model must perform better than {baseline_score}% in order to predict more than just the plurality class.')

# # CNN Model 2: Data Augmentation

# Since my original dataset is relatively small, with only a little over 1,000 images, I'll do some Keras magic to trick my model into think I have a larger, more varied dataset. Keras' `ImageDataGenerator` class generates "batches of tensor image data with real-time data augmentation. The data will be looped over (in batches)." What that means is when I `fit` my model and run my images through the model, `ImageDataGenerator` will alter my images according to my specifications and run those altered images through my model. Here are the alterations I made to my images:
#  - `rotation_range`: (int) rotates images
#  - `width_shift_range`: (float) makes image wider horizontally
#  - `height_shift_range`: (float) makes image longer vertically
#  - `zoom_range`: (float) randomly zooms in
#  - `shear_range`: (float) randomly applies [shear mapping](https://en.wikipedia.org/wiki/Shear_mapping)
#  - `horizontal_flip`: (boolean) applies horizontal flip
#  - `fill_mode`: (`"constant"`, `"nearest"`, `"reflect"` or `"wrap"`) fills newly created pixels


# Code modified from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Set parameters to modify images
datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             shear_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Sample one image from original dataset
x = X[40:41]

# Reshape it to (1, 110, 75, 1)
x = x.reshape(x.shape + (1,))

# The .flow() command below generates batches of randomly transformed images
plt.title("Data Augmentation Sample", fontsize=14)
i = 0
for im in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(im[0]), cmap=plt.cm.gray)
    i += 1
    if i % 3 == 0:
        break

plt.show()

# [Image Source](https://www.flickr.com/photos/istolethetv/14547691835/in/photolist-oawHX2-MoQjvg-6JGaA5-HYdVtW-q1eLJd-eAgHvG-p4ndog-q16Pjx-ZnewPd-pHFi3Z-eAeqyP-9UZu2w-7kq24n-aiTZTf-gqi1T-pHHmuy-kYe27-2LURLm-8VTg3Y-6p68bA-qovm-Kz3Xy-pHHreU-dYZZGa-AKSRe-8dVuHW-duDNw2-aiTZph-d6ekGm-WNRTKn-p4jiCS-7mTWPN-pHLAZw-eAigCL-q168Uv-a7DxQd-i2dZo4-nVizxY-6JBVbB-gp1Gm-HYdHD3-p4njnR-nQG9z5-asfFGj-62LwUH-hsFKq-4rTTJh-cKzJK5-pY1cDY-d6efmY)

# ## Design Neural Network


# Instantiate Convolutional Neural Network
cnn_model_2 = Sequential()

# First conv module: input layer
cnn_model_2.add(Conv2D(filters=32,
                       kernel_size=(3, 3),
                       activation='relu',
                       input_shape=X_train[0].shape))

cnn_model_2.add(MaxPooling2D(pool_size=(2, 2)))
print("module 1")

# Second conv module
cnn_model_2.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn_model_2.add(MaxPooling2D(pool_size=(2, 2)))
print("module 2")

# Third conv module
cnn_model_2.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn_model_2.add(MaxPooling2D(pool_size=(2, 2)))
print("module 3")

# Fourth conv module
cnn_model_2.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn_model_2.add(MaxPooling2D(pool_size=(2, 2)))
print("module 4")

# Flatten layer to pass through dense layer
cnn_model_2.add(Flatten())
print("flatten")

# Dropout layer to avoid overfitting
cnn_model_2.add(Dropout(rate=1 - 0.5))
print("dropout")

# Densely connected layer
cnn_model_2.add((Dense(512, activation='relu')))
print("relu")

# Output layer
cnn_model_2.add(Dense(y_test.shape[1], activation='softmax'))
print("softmax")

# ### Set parameters for data augmentation

# Here, I apply `ImageDataGenerator` to augment my training dataset. I do not need to augment my testing dataset since the training dataset is what will be used to train my model.


# Set batch size
batch_size = 64

# This is the augmentation configuration to use for training
train_datagen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Augment training data
print("Augment training data")
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

# I don't need to do anything to my testing data since it's already been scaled

# Check shape of ImageDataGenerator images
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# ### Compile the model


# Compile model
print("Starting to compile")
cnn_model_2.compile(loss='categorical_crossentropy',  # Loss function for multiclassification
                    optimizer='adam',
                    metrics=['accuracy'])

# Save model
print("Save model")
cnn_model_2.save('./data/model_fashion_2.h5')
