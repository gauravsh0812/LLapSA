import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests

# Load the pre-trained SigLIP model and its processor
model_name = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    if image_path.startswith("http"):  # If image is from a URL
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:  # If image is from a local file
        image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image using the model's processor
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

# Function to extract features from the image
def extract_features(image_path):
    # Preprocess the image
    pixel_values = preprocess_image(image_path)
    
    # Extract features using the model
    with torch.no_grad():
        features = model(pixel_values=pixel_values).last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
    
    return features

# Example usage
image_path = "https://example.com/sample.jpg"  # Replace with your image URL or local path
features = extract_features(image_path)

# Output the shape of the features
print("Extracted Features Shape:", features.shape)
