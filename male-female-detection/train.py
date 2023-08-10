# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import tensorflow as tf

# Set initial parameters
epochs = 50
lr = 1e-3j  # Learning rate (controls how quickly the model learns)
batch_size = 64  # Number of images to process at once
img_dims = (96, 96, 3)  # Dimensions of input images (width, height, color channels)

# Create empty lists to store data and labels
data = []  # List to hold image data
labels = []  # List to hold image labels (gender)

# Load image files from the dataset folder
image_files = [f for f in glob.glob(r'./gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)  # Shuffle the order of image files randomly

# Loop through each image file
for img in image_files:
    # Read the image
    image = cv2.imread(img)
    
    # Resize the image to the desired dimensions
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    
    # Convert the image to an array
    image = img_to_array(image)
    
    # Append the image data to the 'data' list
    data.append(image)

    # Extract label from the file path
    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1  # Assign 1 for female
    else:
        label = 0  # Assign 0 for male
        
    # Append the label to the 'labels' list
    labels.append([label])

# Convert lists to numpy arrays
data = np.array(data, dtype="float") / 255.0  # Normalize pixel values to range between 0 and 1
labels = np.array(labels)

# Split the dataset into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to categorical format
trainY = to_categorical(trainY, num_classes=2)  # Convert gender labels to one-hot encoded format
testY = to_categorical(testY, num_classes=2)

# Create an image data generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Define a function to build the model architecture
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    # Add convolutional and pooling layers to the model
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # Add more convolutional layers
    # ... (similar lines repeated for different layers)

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Add the output layer for gender classification
    model.add(Dense(classes))
    model.add(Activation("sigmoid"))  # Output is a probability for each gender

    return model

# Build the model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Set up a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr, decay_steps=len(trainX) // batch_size, decay_rate=0.9
)
# Choose an optimizer and compile the model
opt = Adam(learning_rate=lr_schedule)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Create an image data generator for training with data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_generator = train_datagen.flow(trainX, trainY, batch_size=batch_size)

# Train the model
H = model.fit(train_generator, validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Save the trained model
model.save('gender_detection.keras')

# Plot training and validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save the plot to a file
plt.savefig('plot.png')
