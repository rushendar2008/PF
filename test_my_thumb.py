import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os
import numpy as np

# 1. Define Model
class FingerprintGenderModel(nn.Module):
    def __init__(self):
        super(FingerprintGenderModel, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.resnet(x)

def preprocess_and_test(image_path):
    print("--- 1. PREPROCESSING REAL PHOTO ---")
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return
        
    # Crop to the center of the thumb to avoid background
    h, w = img.shape
    crop_img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    
    # Enhance contrast (Equalization)
    eq_img = cv2.equalizeHist(crop_img)
    
    # Adaptive thresholding to extract ridges (making it look like a B&W scan)
    binary_img = cv2.adaptiveThreshold(eq_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 5)
    
    # Save the processed image so we can see what the AI is actually looking at
    processed_path = "processed_thumb.jpg"
    cv2.imwrite(processed_path, binary_img)
    print(f"Converted your optical photo into a fingerprint scan: {processed_path}")
    
    print("\n--- 2. RUNNING AI BRAIN ---")
    model = FingerprintGenderModel()
    model.load_state_dict(torch.load("fingerprint_gender_model.pth", map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(processed_path).convert('RGB') # ResNet expects 3 channels
    image_tensor = transform(image).unsqueeze(0) 

    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        
    print("\n--- NEW AI Result ---")
    if probability < 0.5:
        print("Prediction: MALE")
        print(f"Confidence: {(1 - probability) * 100:.1f}% sure it is Male.")
    else:
        print("Prediction: FEMALE")
        print(f"Confidence: {probability * 100:.1f}% sure it is Female.")

if __name__ == "__main__":
    preprocess_and_test("thumb.jpg")
