import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
train_data = pd.read_csv("dataset\sign_mnist_train\sign_mnist_train.csv")  # Correct path for training data
test_data = pd.read_csv("dataset\sign_mnist_test\sign_mnist_test.csv")  # Correct path for testing data

# Extract labels and pixel values
train_labels = train_data['label'].values
test_labels = test_data['label'].values
train_images = train_data.iloc[:, 1:].values
test_images = test_data.iloc[:, 1:].values

# Normalize pixel values to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to 28x28x1 (as grayscale images)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(25, activation='softmax')
])

# Compile the model with sparse_categorical_crossentropy loss and 25 classes
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Display the model summary
model.summary()

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

# Save the trained model to disk
model.save("sign_language_model1.keras")
