import os
import numpy as np
import cv2
import torch
from skimage.segmentation import slic
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Constants
NUM_CLASSES = 5  # Adjust based on the number of roof material types
DEVICE = torch.device("cpu")  # Use "cuda" for GPU usage
SATELLITE_FOLDER = "./images/satellites"
MASK_FOLDER = "./images/masks"
MODEL_PATH = "./gcn_roof_model.pth"

# GCN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Preprocess a single satellite image into a graph
def preprocess_data(image_path):
    # Load satellite image
    image = cv2.imread(image_path)
    
    # Segment satellite image into superpixels
    segments = slic(image, n_segments=100, compactness=10, start_label=1)
    
    # Generate node features (average color of each superpixel)
    node_features = []
    unique_segments = np.unique(segments)

    for segment in unique_segments:
        mask_segment = segments == segment
        avg_color = image[mask_segment].mean(axis=0)
        node_features.append(avg_color)
    
    # Convert to PyTorch tensor
    node_features = torch.tensor(np.array(node_features), dtype=torch.float32).to(DEVICE)
    
    # Generate simple edge connections (adjacent superpixels)
    edges = [[i, i + 1] for i in range(len(unique_segments) - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

    # Create the graph data object
    data = Data(x=node_features, edge_index=edge_index)
    data.segments = segments  # Optional: useful if you want to map predictions back to the original image shape
    return data

# Load the trained model from a checkpoint
def load_checkpoint(path, model):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from: {path}.")
        return model
    else:
        print(f"No checkpoint found at: {path}.")
        return model

# Predict using the trained GCN model
def predict(model, satellite_image_path):
    model.eval()
    # Preprocess the image into a graph
    data = preprocess_data(satellite_image_path)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        predictions = output.argmax(dim=1)  # Choose the class with the highest probability for each node

    return predictions, data.segments


# Define a color palette for the classes (you can adjust these colors as needed)
CLASS_COLORS = [
    [0, 0, 0],       # Class 0: Black (background or empty)
    [255, 0, 0],     # Class 1: Red
    [0, 255, 0],     # Class 2: Green
    [0, 0, 255],     # Class 3: Blue
    [255, 255, 0]    # Class 4: Yellow
]


def save_predictions(image_path, predictions, segments, output_folder, alpha=0.5):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the input image
    image = cv2.imread(image_path)
    
    # Prepare the prediction map and segmented image
    segmented_image = np.zeros_like(image)
    prediction_map = np.zeros_like(segments)
    
    # Loop through each unique segment
    unique_segments = np.unique(segments)
    for i, segment in enumerate(unique_segments):
        mask_segment = segments == segment
        predicted_class = predictions[i].item()  # Get the predicted class for the superpixel
        
        # Assign the predicted class to the corresponding pixels in the prediction map
        prediction_map[mask_segment] = predicted_class

        # Assign the color corresponding to the predicted class (using the CLASS_COLORS dictionary)
        color = CLASS_COLORS[predicted_class]  # Get the color for the predicted class
        segmented_image[mask_segment] = color  # Apply the color to the segment
    
    # Prepare the transparent overlay (blend the original image with the prediction)
    blended_image = cv2.addWeighted(image, 1 - alpha, segmented_image, alpha, 0)

    # Prepare the file path for saving the output image
    base_name = os.path.basename(image_path)
    file_name, _ = os.path.splitext(base_name)
    output_path = os.path.join(output_folder, f"{file_name}_predicted.png")

    # Save the resulting image
    cv2.imwrite(output_path, blended_image)
    print(f"Prediction saved as: {output_path}")

model = GCN(in_channels=3, hidden_channels=16, out_channels=NUM_CLASSES).to(DEVICE)
model = load_checkpoint(MODEL_PATH, model)

# Process a set of images and make predictions
satellite_images = sorted([f for f in os.listdir(SATELLITE_FOLDER) if f.lower().endswith('.png')])

for image_name in satellite_images:
    image_path = os.path.join(SATELLITE_FOLDER, image_name)
    
    # Make prediction
    predictions, segments = predict(model, image_path)
    
    # Visualize the results
    save_predictions(image_path, predictions, segments, "./images/val_outputs/")