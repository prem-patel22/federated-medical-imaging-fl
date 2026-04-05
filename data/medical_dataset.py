import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from collections import Counter
import os

def get_medical_dataset():
    """
    Use COVID-19 Radiography Dataset (works directly, no download issues)
    """
    print("📥 Loading COVID-19 Radiography Dataset...")
    
    # Transform for medical images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download COVID-19 dataset (from TorchVision)
    try:
        # Alternative 1: COVID-19 dataset via torchvision
        dataset = torchvision.datasets.ImageFolder(
            root='./data/raw', 
            transform=transform
        )
        print("✅ Dataset loaded from existing files")
    except:
        # If not available, create a synthetic medical dataset for testing
        print("⚠️ Creating synthetic medical dataset for testing...")
        dataset = create_synthetic_medical_dataset(transform)
    
    return dataset

def create_synthetic_medical_dataset(transform):
    """
    Create synthetic medical images (for testing when real data isn't available)
    """
    from PIL import Image
    import torch.utils.data as data
    
    class SyntheticMedicalDataset(data.Dataset):
        def __init__(self, num_samples=1000, img_size=224, num_classes=3, transform=None):
            self.num_samples = num_samples
            self.img_size = img_size
            self.num_classes = num_classes
            self.transform = transform
            
            # Generate random images and labels
            self.images = torch.randn(num_samples, 3, img_size, img_size)
            self.labels = torch.randint(0, num_classes, (num_samples,))
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            # Convert to PIL for transforms
            image = image.permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype('uint8')
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    dataset = SyntheticMedicalDataset(num_samples=2000, transform=transform)
    return dataset

def prepare_hospital_data():
    """
    Prepare data for 3 hospitals with different distributions
    """
    print("\n🏥 Creating Hospital Data Partitions...")
    
    # Get dataset
    dataset = get_medical_dataset()
    
    # Get class names
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    print(f"📊 Classes: {class_names}")
    print(f"📊 Total samples: {len(dataset)}")
    
    # Split into train (70%), validation (15%), test (15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\n📊 Split sizes:")
    print(f"   Training: {train_size} samples")
    print(f"   Validation: {val_size} samples")
    print(f"   Test: {test_size} samples")
    
    # Create non-IID partitions for 3 hospitals
    num_clients = 3
    alpha = 0.5  # Higher non-IID
    
    # Get labels from train dataset
    if hasattr(train_dataset, 'indices'):
        # For Subset dataset
        labels = []
        for idx in train_dataset.indices:
            if hasattr(dataset, 'targets'):
                labels.append(dataset.targets[idx])
            else:
                labels.append(dataset[idx][1])
    else:
        labels = [dataset[i][1] for i in range(len(train_dataset))]
    
    labels = np.array(labels)
    num_classes = len(class_names)
    
    # Generate Dirichlet distribution for non-IID split
    client_data_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        
        if len(class_indices) == 0:
            continue
            
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        assigned = 0
        for client_id, prop in enumerate(proportions):
            num_samples = int(prop * len(class_indices))
            if client_id == num_clients - 1:
                num_samples = len(class_indices) - assigned
            
            if num_samples > 0:
                start = assigned
                end = assigned + num_samples
                client_data_indices[client_id].extend(class_indices[start:end])
                assigned += num_samples
    
    # Create datasets for each hospital
    hospital_datasets = []
    for indices in client_data_indices:
        if len(indices) > 0:
            hospital_datasets.append(Subset(train_dataset, indices))
        else:
            # Empty dataset fallback
            hospital_datasets.append(Subset(train_dataset, []))
    
    # Create data loaders
    hospital_loaders = []
    for i, h_dataset in enumerate(hospital_datasets):
        if len(h_dataset) > 0:
            loader = DataLoader(h_dataset, batch_size=32, shuffle=True)
        else:
            loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        hospital_loaders.append(loader)
        
        # Print distribution
        print(f"\n🏥 Hospital {chr(65+i)} (Client {i+1}):")
        print(f"   Total samples: {len(h_dataset)}")
        
        # Show class distribution
        if len(h_dataset) > 0:
            h_labels = []
            for idx in range(len(h_dataset)):
                if hasattr(h_dataset, 'indices'):
                    orig_idx = h_dataset.indices[idx]
                    if hasattr(train_dataset, 'indices'):
                        final_idx = train_dataset.indices[orig_idx]
                        if hasattr(dataset, 'targets'):
                            h_labels.append(dataset.targets[final_idx])
                else:
                    h_labels.append(h_dataset[idx][1])
            
            if h_labels:
                distribution = Counter(h_labels)
                for class_id, count in distribution.items():
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    print(f"   {class_name}: {count} ({count/len(h_dataset)*100:.1f}%)")
    
    # Create validation and test loaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return hospital_loaders, val_loader, test_loader, class_names

if __name__ == "__main__":
    print("="*50)
    print("🏥 Federated Medical Imaging - Data Setup")
    print("="*50)
    
    hospital_loaders, val_loader, test_loader, class_names = prepare_hospital_data()
    
    print("\n" + "="*50)
    print("✅ Data preparation complete!")
    print("="*50)
    print(f"\n📊 Summary:")
    print(f"   - 3 Hospitals with non-IID data distributions")
    print(f"   - {len(class_names)} classes: {class_names}")
    print(f"   - Ready for Federated Learning!")