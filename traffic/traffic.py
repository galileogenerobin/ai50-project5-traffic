import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # The required shape for our image data
    shape = (IMG_WIDTH, IMG_HEIGHT)

    # Read data from each subdirectory
    for subdirectory in range(NUM_CATEGORIES):
        subdirectory_path = os.path.join(data_dir, str(subdirectory))
        for file in os.listdir(subdirectory_path):
            # Convert image data into numpy ndarray
            img = cv2.imread(os.path.join(subdirectory_path, file))
            # Resize to the required shape for the analysis
            img = cv2.resize(img/255, shape)
            # Populate list of images
            images.append(img)
            # Populate list of labels
            labels.append(subdirectory)

    # Return tuple
    return (images, labels)
    # raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Experimentation variables
    conv2d_filters = 32
    conv2d_size = (3, 3)
    pool_layer_size = (2, 2)
    hidden_layer_size = 256
    dropout_value = 0.5
    shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    # Create a new neural network model
    model = tf.keras.models.Sequential([
        # Conv2D layer
        tf.keras.layers.Conv2D(
            conv2d_filters, conv2d_size, activation="relu", input_shape=shape
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=pool_layer_size),
        
        # Conv2D layer
        tf.keras.layers.Conv2D(
            conv2d_filters, conv2d_size, activation="relu"
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=pool_layer_size),
    
        # # Conv2D layer
        # tf.keras.layers.Conv2D(
        #     conv2d_filters, conv2d_size, activation="relu"
        # ),
        # # Max-pooling layer
        # tf.keras.layers.MaxPooling2D(pool_size=pool_layer_size),

        # Flatten
        tf.keras.layers.Flatten(),

        # Hidden layer(s)
        tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
        # tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
        # tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
        # tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
        # tf.keras.layers.Dense(hidden_layer_size, activation="relu"),

        # Dropout
        tf.keras.layers.Dropout(dropout_value),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # print(model.summary())

    return model
    # raise NotImplementedError


if __name__ == "__main__":
    main()
