import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Parent folder containing the augmented images
parent_folder = "preprocessed_images"

# Subfolders representing the classes
classes = []
for entry in os.scandir(parent_folder):
    if entry.is_dir():
        classes.append(entry.name)

# Image dimensions
image_width = 224
image_height = 224

# Load and preprocess the augmented images
data = []
labels = []

for class_index, class_folder in enumerate(classes):
    class_path = os.path.join(parent_folder, class_folder)
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        data.append(image)
        labels.append(class_index)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Normalize the pixel values between 0 and 1
train_data = train_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# Convert the labels to one-hot encoding
num_classes = len(classes)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Define the CNN model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), activation="relu", input_shape=(image_width, image_height, 3))
)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_data,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_data, test_labels),
)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Save the model architecture as JSON
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("model_weights.h5")

# Save the class labels
class_labels = {}
for i in range(len(classes)):
    class_labels[f"{i}"] = classes[i]
np.save("class_labels.npy", class_labels)
