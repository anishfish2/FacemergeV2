import numpy as np
import sklearn 
import cv2 
from sklearn.decomposition import PCA
import os 

# Specify the directory path
directory = "aligned/"

# Get a list of all files in the directory
files = os.listdir(directory)

# Filter out only the image files
image_files = [file for file in files if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]

# Print the list of image files
images = []

for file in image_files:
    images.append(cv2.imread(directory + file))

combined_img = np.vstack([i.reshape(-1) for i in images])

n_components = len(image_files)

pca = PCA(n_components=n_components)

# Fit PCA on the combined images
pca.fit(combined_img)

# Transform the images into the latent space
img1_latent = pca.transform(img1_flat.reshape(1, -1))


# Compute the average of the latent representations
average_latent = (img1_latent + img2_latent) / 2

# Reconstruct the average latent representation back into the original space
average_reconstructed = pca.inverse_transform(average_latent)

# Reshape the reconstructed average image back to its original shape
average_reconstructed = average_reconstructed.reshape(360, 360, 3)

# Convert the average reconstructed image to uint8 data type (required by OpenCV)
average_reconstructed_uint8 = average_reconstructed.astype(np.uint8)

# Specify the file path where you want to save the image
output_path = "merged/average_image.png"

# Save the image using OpenCV
cv2.imwrite(output_path, average_reconstructed_uint8)

print(f"Average image saved as {output_path}")