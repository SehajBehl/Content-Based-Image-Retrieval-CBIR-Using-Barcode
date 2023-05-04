import os
import numpy as np
from PIL import Image

# Set the path to the MNIST_DS folder
mnist_path = './MNIST_DS'

# Create an empty list to store the barcodes
barcodes = []

# Loop through the subfolders for each digit
for digit in range(10):
    # Set the path to the subfolder for the current digit
    digit_path = os.path.join(mnist_path, str(digit))

    # Loop through the images in the subfolder
    for image_name in os.listdir(digit_path):
        # Set the path to the current image
        image_path = os.path.join(digit_path, image_name)

        # Load the image using Pillow
        image = Image.open(image_path)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Create the projections for each angle
        p1 = np.sum(image_array, axis=0)
        p2 = np.sum(np.diag(image_array))[np.newaxis]
        p3 = np.sum(image_array, axis=1)
        p4 = np.sum(np.diag(np.fliplr(image_array)))[np.newaxis]

        # Calculate the threshold values for each projection
        th_p1 = np.mean(p1)
        th_p2 = np.mean(p2)
        th_p3 = np.mean(p3)
        th_p4 = np.mean(p4)

        # Create the barcode by concatenating the projections and assigning 0 or 1 based on the threshold values
        barcode = np.concatenate([
            (p1 <= th_p1).astype(int),
            (p2 <= th_p2).astype(int),
            (p3 <= th_p3).astype(int),
            (p4 <= th_p4).astype(int)
        ])

        # Add the barcode to the list of barcodes
        barcodes.append(barcode)

# Convert the list of barcodes to a NumPy array
barcodes = np.array(barcodes)


# Define a function to calculate the Hamming distance between two barcodes
def hamming_distance(barcode1, barcode2):
    return np.sum(barcode1 != barcode2)


# Define a function to find the most similar image in the dataset using the Hamming distance
def find_most_similar(query_barcode):
    distances = [hamming_distance(query_barcode, barcode) for barcode in barcodes]
    return np.argmin(distances)


# Example usage: Find the most similar image to the first image in the dataset
most_similar_index = find_most_similar(barcodes[1])
print(f'The most similar image is at index {most_similar_index}')
