import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import kagglehub

# 1. Download the Real Dataset
print("Downloading SOCOFing Dataset...")
dataset_path = kagglehub.dataset_download("ruizgara/socofing")
print(f"Dataset downloaded to: {dataset_path}")

# In the SOCOFing dataset, the actual real fingerprints are usually in a folder called 'Real'
real_images_dir = os.path.join(dataset_path, "SOCOFing", "Real")
if not os.path.exists(real_images_dir):
    # Fallback in case the folder structure is slightly different
    real_images_dir = dataset_path 

# 2. Define the Real Dataset Loader
class RealFingerprintDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.genders = []
        
        # Preprocessing: Resize to 224x224 and normalize for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("Parsing real fingerprint files...")
        for file in os.listdir(root_dir):
            if file.lower().endswith('.bmp'):
                # Example filename: "100__M_Left_index_finger.BMP"
                parts = file.split('__')
                if len(parts) > 1:
                    traits = parts[1].split('_')
                    gender_str = traits[0]  # 'M' or 'F'
                    
                    # 0.0 for Male, 1.0 for Female
                    gender_label = 0.0 if gender_str == 'M' else 1.0 
                    
                    self.image_paths.append(os.path.join(root_dir, file))
                    self.genders.append(gender_label)
                    
        print(f"Successfully loaded {len(self.image_paths)} real fingerprints!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        gender = torch.tensor([self.genders[idx]], dtype=torch.float32)
        return image, gender

# 3. Define the AI Model (Gender Only)
class FingerprintGenderModel(nn.Module):
    def __init__(self):
        super(FingerprintGenderModel, self).__init__()
        # Load a pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace the final layer to output a single value (Gender Probability)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() # Outputs a probability between 0 and 1
        )

    def forward(self, x):
        return self.resnet(x)

# 4. Training Loop
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    dataset = RealFingerprintDataset(real_images_dir)
    
    # Split into 80% Training and 20% Testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model, Loss Function, and Optimizer
    model = FingerprintGenderModel().to(device)
    save_path = "fingerprint_gender_model.pth"
    
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path} to train further...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Starting training from scratch...")
        
    criterion = nn.BCELoss() # Binary Cross Entropy for Male/Female
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i, (images, genders) in enumerate(train_loader):
            images, genders = images.to(device), genders.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, genders)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress every 10 batches
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Complete. Average Loss: {total_loss/len(train_loader):.4f}")

    print("\nTraining Complete! The model has now learned from real fingerprints.")
    
    # Save the trained model weights
    save_path = "fingerprint_gender_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}!")

if __name__ == '__main__':
    train_model()
