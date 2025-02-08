import os
import torch
import csv
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Get the current working directory
cwd = Path(__file__).parent

# Define the path to ImageNet validation set
imagenet_path = cwd / "imagenet" / "val"

# Load pre-trained ResNet18 model and remove the final classification layer
resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
modules = list(resnet18.children())[:-1]  # Remove final classification layer
model = torch.nn.Sequential(*modules)

# Freeze model parameters to prevent training updates
for p in model.parameters():
    p.requires_grad = False

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256 pixels on the smaller side
    transforms.ToTensor(),  # Convert image to tensor
])

# Define output CSV file path
output_csv = cwd / "image_features.csv"

# Read and process images, then save the results to CSV
with torch.no_grad(), open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=";")  # Use ";" as separator
    writer.writerow(["Image Path", "Feature Vector"])  # Write header

    image_paths = sorted(imagenet_path.glob("**/*.JPEG"))  # Search for all .JPEG images recursively

    for imagepath in image_paths:
        print(f"Processing: {imagepath}")

        # Open image, convert to RGB (to handle grayscale images), and preprocess
        input_tensor = preprocess(Image.open(imagepath).convert("RGB")).to(device).unsqueeze(0)

        # Extract features from the image using the ResNet18 feature extractor
        output = model(input_tensor).squeeze().cpu().numpy().tolist()  # Convert tensor to a Python list

        # Convert feature vector to a JSON-encoded string
        feature_str = json.dumps(output)

        # Write image path and JSON-encoded feature vector to CSV
        writer.writerow([str(imagepath), feature_str])

    print(f"Feature extraction completed. Results saved in {output_csv}")
