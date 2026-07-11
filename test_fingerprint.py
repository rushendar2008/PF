import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# 1. We must define the exact same model architecture so PyTorch knows how to load the weights
class FingerprintGenderModel(nn.Module):
    def __init__(self):
        super(FingerprintGenderModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False) # We don't need pretrained weights from internet anymore
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

def test_single_image(image_path, model_path="fingerprint_gender_model.pth"):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}. Did you run the training script?")
        return

    # 2. Load the model and the trained weights
    print("Loading trained AI brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintGenderModel().to(device)
    
    # Load the weights from the file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode (turns off dropout etc.)
    print("Brain loaded successfully!\n")

    # 3. Load and preprocess the image exactly like we did during training
    print(f"Analyzing fingerprint: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Add a batch dimension [1, 3, 224, 224]
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 4. Make the Prediction!
    with torch.no_grad(): # We don't need gradients for testing
        prediction = model(input_tensor)
        prob = prediction.item() # Get the raw probability score
        
    print("\n--- AI Result ---")
    if prob > 0.5:
        # Remember: 0.0 was Male, 1.0 was Female in our training script
        print(f"Prediction: FEMALE")
        print(f"Confidence: {prob * 100:.1f}% sure it is Female.")
    else:
        print(f"Prediction: MALE")
        print(f"Confidence: {(1.0 - prob) * 100:.1f}% sure it is Male.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_fingerprint.py <path_to_fingerprint_image>")
        print("Example: python test_fingerprint.py sample.bmp")
    else:
        test_single_image(sys.argv[1])
