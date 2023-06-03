import os
import cv2
import numpy as np

# source folder and output folder
image_folder = "images/"
output_folder_parent = "preprocessed_images/"

folders = []
for entry in os.scandir(image_folder):
    if entry.is_dir():
        folders.append(entry.name)

# New dimensions for the resized images
image_size = (224, 224)

# Augmentation parameters
rotation_range = 20  # Range of random rotations (-rotation_range to +rotation_range)
flip_prob = 0.5  # Probability of random horizontal flipping
brightness_range = [0.7, 1.3]  # Range of random brightness adjustment
contrast_range = [0.8, 1.2]  # Range of random contrast adjustment

# Iterate through sub folders
for folder in folders:
    # Path to the current subfolder
    subfolder_path = os.path.join(image_folder, folder)

    # Create a new subfolder for the preprocessed images
    output_folder = os.path.join(output_folder_parent, folder)
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    # Iterate through images in the current subfolder
    for filename in os.listdir(subfolder_path):
        # Load the image
        print(f"i {i}")
        image_path = os.path.join(subfolder_path, filename)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, image_size)

        # normalize resized image
        # normalized_image = resized_image / 255.0
        augmented_images = [resized_image]

        # Random rotation
        angle = np.random.uniform(-rotation_range, rotation_range)
        rotation_matrix = cv2.getRotationMatrix2D(
            (image_size[0] / 2, image_size[1] / 2), angle, 1.0
        )
        rotated_image = cv2.warpAffine(resized_image, rotation_matrix, image_size)
        augmented_images.append(rotated_image)

        # Random horizontal flip
        if np.random.random() < flip_prob:
            flipped_image = cv2.flip(resized_image, 1)
            augmented_images.append(flipped_image)

        # Random brightness adjustment
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        brightness_adjusted_image = np.clip(
            resized_image * brightness_factor, 0, 255
        ).astype(np.uint8)
        augmented_images.append(brightness_adjusted_image)

        # Random contrast adjustment
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        contrast_adjusted_image = np.clip(
            resized_image * contrast_factor, 0, 255
        ).astype(np.uint8)
        augmented_images.append(contrast_adjusted_image)

        # Save the preprocessed image
        j = 0
        for augmented_image in augmented_images:
            output_path = os.path.join(
                output_folder, f'{folder}_{i + (j+1)}.{filename.split(".").pop()}'
            )
            cv2.imwrite(output_path, augmented_image)
            print(f"Augmented image saved: {output_path}")
            j = j + 1
        i = i + j
