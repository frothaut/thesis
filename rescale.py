import os
from PIL import Image

# Path to the folder containing images
folder_path = './images/satellites'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image (you can modify this based on your specific image formats)
    if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        try:
            # Open the image
            with Image.open(file_path) as img:
                # Check if the image is 500x500
                if img.size == (500, 500):
                    # Resize the image to 512x512
                    img_resized = img.resize((512, 512))
                    
                    # Save the resized image (you can overwrite or save with a new name)
                    img_resized.save(file_path)
                    print(f"Resized {filename} to 512x512.")
                else:
                    print(f"Image {filename} is not 500x500, skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")