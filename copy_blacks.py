from PIL import Image
import os

# Path to the folder containing images
folder_path = './images2/satellites'
outpath = './images2/masks'
# Iterate over each file in the directory
for filename in os.listdir(folder_path):
    # Only process image files (check extension)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Full path to the image
        img_path = os.path.join(folder_path, filename)
        
        # Open the image
        with Image.open(img_path) as img:
            # Create a black image with the same size and mode as the original image
            black_img = Image.new('RGB', img.size, color=(0, 0, 0))  # 'RGB' for color images, 'L' for grayscale
            
            # Save the black image with the same name plus '_mask' suffix
            new_filename = f"{os.path.splitext(filename)[0]}_mask{os.path.splitext(filename)[1]}"
            new_img_path = os.path.join(outpath, new_filename)
            black_img.save(new_img_path)

            print(f"Created mask: {new_filename}")