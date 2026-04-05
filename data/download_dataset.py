import os
import requests
import zipfile
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def download_mednist(data_dir='./data/raw'):
    """
    Download MedNIST dataset (small, perfect for testing)
    """
    print("📥 Downloading MedNIST dataset...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # MedNIST URL (from TensorFlow dataset)
    url = "https://github.com/arjun-khandelwal/MedNIST-Dataset/raw/master/MedNIST.zip"
    zip_path = os.path.join(data_dir, "MedNIST.zip")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(len(data))
    
    # Extract
    print("📂 Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.remove(zip_path)
    print("✅ Dataset downloaded and extracted!")
    
    return os.path.join(data_dir, "MedNIST")

def load_mednist_dataset(data_path, img_size=224, batch_size=32):
    """
    Load MedNIST dataset with transforms
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),  # Augmentation
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Get class names
    class_names = dataset.classes
    print(f"📊 Classes: {class_names}")
    print(f"📊 Total images: {len(dataset)}")
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Download and test
    data_path = download_mednist()
    train_loader, val_loader, test_loader, classes = load_mednist_dataset(data_path)
    
    print(f"\n✅ Dataset ready!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")