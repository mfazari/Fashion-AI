# Basic libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Image-specific libraries 
from PIL import Image

np.random.seed(2019)

# # Image Pre-Processing

# Run the function on my images folders
fashion_list = ['caps', 'pants', 'shoes', 'sweater', 't_shirt']


# Function to resize images
def resize_images(images_directory):
    for i, image in enumerate(os.listdir(images_directory)):
        if image != '.DS_Store':  # Passes a hidden file called ".DS_Store"
            image_dir = images_directory + '/' + image
            img = Image.open(image_dir)
            width, height = img.size  # Stores image height and width as variables

            # Check if image is 75 x 110
            if (width == 75) & (height == 110):  # If images meet hxw requirement, pass
                None
            elif (width != 75) or (height != 110):
                img = img.resize((75, 110))  # Reshape images that are not 75 x 110
                img.save(image_dir, optimize=True)  # Save image

    print(f'Your images have been scaled to 75x100!')


for piece in fashion_list:
    fashion_directory = './dataset/' + piece
    resize_images(fashion_directory)


# # Convert images to grayscale and numpy array

# This function will find my images in my directory and convert any colored images to grayscale, and it will convert
# all images to numpy arrays.


# Function to convert images to grayscale and numpy array
def get_array(piece, images_directory):
    array_list = []
    for image in os.listdir(images_directory):
        if image != '.DS_Store':  # Passes a hidden file called ".DS_Store"
            image_dir = images_directory + '/' + image
            img = Image.open(image_dir).convert('L')  # Converts image to grayscale
            img.save(image_dir, optimize=True)

    for image in os.listdir(images_directory):
        if image.endswith(".jpg"):
            image_dir = images_directory + '/' + image
            img = Image.open(image_dir)
            pix = np.array(img)  # Converts image to numpy array
            array_list.append(pix)
    print(f'{str(piece)} shape: {np.stack(array_list).shape}')
    return np.stack(array_list)


# Run function on all pieces.
caps = get_array('caps', './dataset/caps')

pants = get_array('pants', './dataset/pants')

shoes = get_array('shoes', './dataset/shoes')

sweater = get_array('sweater', './dataset/sweater')

t_shirt = get_array('t_shirt', './dataset/t_shirt')

# Create variable for all pieces arrays
piece_arrays = caps, pants, shoes, sweater, t_shirt


# Function to create labels array
def map_pieces(pieces):
    lst = []
    for index, piece in enumerate(pieces):
        for i in range(len(piece)):
            lst.append(index)  # Appends the same number to a list for each mention

    stacked_array = np.stack(lst)  # Creates one array with all image labels (numbers 0 - 5)
    print(f'There are {stacked_array.shape} images in this dataset!')
    return stacked_array


# Call function 
piece_labels = map_pieces(piece_arrays)
print("-" * 40)
print(f'`piece_labels` looks like this: {piece_labels}')

# ## Visualize images distribution

# Data analysis


# Convert array to DataFrame
piece_labels_df = pd.DataFrame(piece_labels, columns=['pieces_labels'])

# Create named labels dictionary
piece_dict = {0: 'caps (caps)',
              1: 'pants (pants)',
              2: 'shoes (shoes)',
              3: 'sweater (sweater)',
              4: 't_shirt (t_shirt)'}

# Map pieces_dict to turn numbers to their pieces strings
piece_labels_df['pieces_labels'] = piece_labels_df['pieces_labels'].map(piece_dict)

# Preview head
piece_labels_df.head()


# Function to make bar plot
def bar_plot(x, y, title, color, filename):
    # Set up barplot
    plt.figure(figsize=(9, 5))
    g = sns.barplot(x, y, color=color)
    ax = g

    # Label the graph
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=10)

    # Code modified from http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for p in ax.patches:
        totals.append(p.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width() + .3, p.get_y() + .38, int(p.get_width()), fontsize=10)
    plt.savefig(f'./images/other/{filename}.png', bbox_inches='tight')


# Visualize position distribution
bar_plot(piece_labels_df['pieces_labels'].value_counts(),
         piece_labels_df['pieces_labels'].value_counts().index,
         "Distribution of Images per piece", 'm', 'image_dataset_distribution')

# Plot images
fig, axes = plt.subplots(3, 3, figsize=(6, 8))
ax = axes.ravel()
plt.suptitle('Clothing pieces', y=1.03, size=16)

ax[0].imshow(caps[20], cmap=plt.cm.gray)
ax[0].set_title("caps")
ax[1].imshow(pants[20], cmap=plt.cm.gray)
ax[1].set_title("pants")
ax[2].imshow(shoes[40], cmap=plt.cm.gray)
ax[2].set_title("shoes")
ax[3].imshow(sweater[1], cmap=plt.cm.gray)
ax[3].set_title("sweater")
ax[4].imshow(t_shirt[2], cmap=plt.cm.gray)
ax[4].set_title("t_shirt")

fig.tight_layout()
plt.show()
fig.savefig(f'./dataset/preview_dataset.png', bbox_inches='tight')

# # Predictor array (all image pixels)


# Concatenate all 5 clothing pieces arrays into one array
X = np.concatenate((piece_arrays))
print(f'Shape of X: {X.shape}')
print("-" * 40)
print(f'Number of images in my dataset: {X.shape[0]}')
print("-" * 40)
print("Dimensions of all images:", X[0].shape)

# Set y variable, pieces labels that correspond to the images
y = piece_labels

# Count of labels 
print(f'Number of image labels: {len(y)}')

# # Save image data to csv


# Get pixel width for flattened image
pixels_flat = X.shape[1] * X.shape[2]
print(f'The pixel width, when flattened, is: {pixels_flat}')

# Save image pixels to DataFrame, one row represents one image
image_pixels_df = pd.DataFrame(X.reshape(len(y), pixels_flat))

# Add labels column
# y = piece_labels
image_pixels_df['piece_num_labels'] = y

# Map pieces to numeric values
image_pixels_df['pieces_string_labels'] = image_pixels_df['piece_num_labels'].map(piece_dict)
print(image_pixels_df)

# Preview head
image_pixels_df.head(3)

# Preview tail
image_pixels_df.tail(3)

# Save image data to csv
image_pixels_df.to_csv('./data/image_pixels.csv')


'''
image_pixels_df:

       0    1    2    3  ...  8248  8249  piece_num_labels  pieces_string_labels
0    238  238  238  238  ...   238   238                 0           caps (caps)
1    250  250  250  250  ...   250   250                 0           caps (caps)
2    255  255  255  255  ...   255   255                 0           caps (caps)
3    234  234  234  234  ...   234   234                 0           caps (caps)
4    235  235  235  235  ...   235   235                 0           caps (caps)
..   ...  ...  ...  ...  ...   ...   ...               ...                   ...
466  232  232  232  232  ...   231   231                 4     t_shirt (t_shirt)
467  236  236  236  236  ...   229   229                 4     t_shirt (t_shirt)
468  235  235  235  235  ...   231   231                 4     t_shirt (t_shirt)
469  231  231  231  231  ...    51    32                 4     t_shirt (t_shirt)
470  235  235  235  235  ...   230   230                 4     t_shirt (t_shirt)

[471 rows x 8252 columns]
'''
