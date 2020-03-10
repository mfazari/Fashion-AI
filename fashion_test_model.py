import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns
import tensorflow as tf
# Model evaluation
from sklearn import metrics

from fashion_create_model import X_test, y_test_copy

cnn_model_2 = tf.keras.models.load_model('./data/model_fashion_2.h5')

# Reset variables
X_test_1 = X_test.copy()

# Undo scale to convert pixels from 0-1 range to 0-255 range
X_test_1 *= 255

# Reshape the X_test since I only need to make predictions on the testing data,
# not the training data
X_test_1 = X_test_1.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Show shape of reshaped X_test_1
print(f'Shape of X_test_1: {X_test_1.shape}')

# Check X_test_1 to see if it's been converted back to pixel range 0-255
X_test_1[0]

# ## Get predicted labels

# We use our trained model 2 (cnn_model_2) to make predictions


# Get predicted classes from unaltered X_test
pred_labels = cnn_model_2.predict_classes(X_test)

# Preview of predicted labels
pred_labels[20:30]

# Create named labels dictionary
pieces_dict = {0: 'caps (caps)',
               1: 'pants (pants)',
               2: 'shoes (shoes)',
               3: 'sweater (sweater)',
               4: 't_shirt (t_shirt)'}

# Convert numbers to associated piece
pred_labels = np.vectorize(pieces_dict.get)(pred_labels)

# Preview predicted labels
pred_labels[20:30]

# ## Get actual labels


# Preview y_test_copy labels
y_test_copy[20:30]

# Convert numbers to associated piece
actual_labels = np.vectorize(pieces_dict.get)(y_test_copy)

# Preview actual labels
actual_labels[20:30]

# ## Get prediction probabilities


# Assign variable to hold all prediction probability results
prediction_proba_array = cnn_model_2.predict(X_test)

# Preview the array. This shows the probabilities that the CNN model predicted
prediction_proba_array[0]

# Save all pieces to a list
piece_list = ['caps (caps)',
              'pants (pants)',
              'shoes (shoes)',
              'sweater (sweater)',
              't_shirt (t_shirt)']

# Convert array to DataFrame
prediction_proba_df = pd.DataFrame(prediction_proba_array, columns=piece_list)

# Preview DataFrame
prediction_proba_df.head(2)


# ## Visualize prediction probabilities of the different pieces


# Function to graph image and probability of predictions
def show_prediction_proba(n):
    # This code block will print interpretation of results
    pred_series = prediction_proba_df.loc[n].sort_values(ascending=False)
    true_perc = round(prediction_proba_df.loc[n][actual_labels[n]] * 100, 2)
    if actual_labels[n] == pred_series.index[0]:
        print("\U0001f600 CORRECT PREDICTION!")
    elif (round(pred_series[0] * 100, 2) - true_perc <= 4) & (actual_labels[n] != pred_series.index[0]):
        print("\U0001F610 INCORRECT PREDICTION, but close call!")
    else:
        print("\U0001F612 INCORRECT PREDICTION!")
    print(f' - Probability of predicting actual piece ({actual_labels[n]}): {true_perc}%')
    message = (f'This piece is a {actual_labels[n]} '
               f', and the model predicted it is showing '
               f'the {pred_series.index[0]} piece. '
               f'The model made the highest prediction at {round(pred_series[0] * 100, 2)}% '
               f'for the piece of {pred_series.index[0]}, '
               f'followed by {round(pred_series[1] * 100, 2)}% for '
               f'for the piece of {pred_series.index[1]}.')
    print()
    print(message)

    # Set parameters for subplot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(wspace=.88)
    ax = axes.ravel()

    # Plot image
    ax[0].imshow(X_test_1[n], cmap=plt.cm.gray)

    # Set image plot title to the actual label
    ax[0].set_title(f'Actual: {actual_labels[n]}', size=13)

    # Plot predicted probabilities distribution
    ax[1].barh(piece_list, prediction_proba_array[n], color='purple')
    ax[1].barh

    # Set distribution plot title to predicted label
    ax[1].set_title(f'Predicted: {pred_labels[n]}', size=13)
    fig.savefig(f'./images/fashion/prediction_{n}.png', bbox_inches='tight');


show_prediction_proba(1)

show_prediction_proba(54)

show_prediction_proba(80)

show_prediction_proba(81)

show_prediction_proba(82)
