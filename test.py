import os
import cv2
import numpy as np

# Folder paths
SATELLITE_FOLDER = "./images/satellites"
MASK_FOLDER = "./images/masks"

# Function to check if a mask is completely or 90% black
def is_black_or_mostly_black(mask_path, threshold=0.95):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
    if mask is None:  # If the mask file cannot be read, return False
        return False
    total_pixels = mask.size
    black_pixels = np.sum(mask == 0)  # Count black pixels
    black_ratio = black_pixels / total_pixels
    return black_ratio >= threshold  # Return True if threshold is met

# Delete masks and corresponding satellite images
def clean_images(satellite_folder, mask_folder):
    masks = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.png')])
    for mask_file in masks:
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Extract base name and construct satellite filename
        mask_basename, mask_ext = os.path.splitext(mask_file)
        satellite_file = f"{mask_basename}_wms{mask_ext}"
        satellite_path = os.path.join(satellite_folder, satellite_file)

        # Check if the mask is black or mostly black
        if is_black_or_mostly_black(mask_path):
            # Delete the mask and its corresponding satellite image
            os.remove(mask_path)
            if os.path.exists(satellite_path):
                os.remove(satellite_path)
            print(f"Deleted: {mask_path} and {satellite_path}")

# Run the cleaning process
clean_images(SATELLITE_FOLDER, MASK_FOLDER)