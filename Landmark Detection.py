import os
import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import gc  # For memory cleanup

# ✅ Download dataset
path = kagglehub.dataset_download("shubhamchauhan22222/mars-landmark-detection")
print("Path to dataset files:", path)

# ✅ List files
dataset_files = os.listdir(path)
print("Dataset files:", dataset_files)

# ✅ Load CSV file
csv_file = os.path.join(path, "train.csv")
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"train.csv not found in {path}. Available files: {dataset_files}")

df = pd.read_csv(csv_file)
df.rename(columns={"FileName": "id", "Class": "label"}, inplace=True)
print("Dataset Head:\n", df.head())

# ✅ Set image directory
data_dir = os.path.join(path, "train")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Images directory not found at {data_dir}")

# ✅ Reduce dataset size (Optional: Use for debugging)
df = df.sample(frac=0.1, random_state=42)  # Use 10% of data to reduce memory load

# ✅ Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# ✅ Image Transformations (Remove normalization for less memory usage)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Custom Dataset Class
class LandmarkDataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id']
        img_file = os.path.join(self.img_path, img_name)

        if not os.path.exists(img_file):
            print(f"Warning: Image {img_file} not found. Skipping.")
            return None, None

        image = Image.open(img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.df.iloc[idx]['label']
        return image, label

# ✅ Create DataLoaders (Reduce batch size & disable multiprocessing)
train_dataset = LandmarkDataset(train_df, data_dir, transform)
val_dataset = LandmarkDataset(val_df, data_dir, transform)

# ✅ Filter out missing images
train_dataset = [data for data in train_dataset if data[0] is not None]
val_dataset = [data for data in val_dataset if data[0] is not None]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# ✅ Load Pretrained Model (CPU Optimized)
device = torch.device("cpu")  # Force CPU mode
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, df['label'].nunique())
model = model.to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop (With Memory Cleanup)
def train_model(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            if images is None or labels is None:
                continue
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")
        
        # ✅ Free unused memory
        gc.collect()
        torch.cuda.empty_cache()

# ✅ Display a batch of images from DataLoader
def show_batch():
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))  # Reduce the number of images displayed
    for i, ax in enumerate(axes.flatten()):
        if i >= len(images):
            break
        img = images[i].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # Unnormalize (optional)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.show()

# ✅ Run training and show images
if __name__ == "__main__":
    show_batch()  # Display sample batch
    train_model(model, train_loader, val_loader)
