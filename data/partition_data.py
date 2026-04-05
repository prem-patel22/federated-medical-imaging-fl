import os
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
from collections import Counter
from download_dataset import download_mednist, load_mednist_dataset

def create_non_iid_partition(dataset, num_clients=3, alpha=0.5):
    """
    Create non-IID partitions using Dirichlet distribution
    alpha=0.5: highly non-IID (realistic for hospitals)
    alpha=100: nearly IID
    """
    num_classes = len(dataset.dataset.classes) if hasattr(dataset, 'dataset') else len(dataset.classes)
    
    # Get labels
    if hasattr(dataset, 'dataset'):
        # For Subset dataset
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets
    
    labels = np.array(labels)
    
    # Generate Dirichlet distribution for each class
    client_data_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(labels == class_id)[0]
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Assign indices to clients
        assigned = 0
        for client_id, prop in enumerate(proportions):
            num_samples = int(prop * len(class_indices))
            if client_id == num_clients - 1:
                num_samples = len(class_indices) - assigned
            
            start = assigned
            end = assigned + num_samples
            
            client_data_indices[client_id].extend(class_indices[start:end])
            assigned += num_samples
    
    # Convert to Subset datasets
    client_datasets = []
    for indices in client_data_indices:
        client_datasets.append(Subset(dataset, indices))
    
    return client_datasets

def prepare_hospital_data():
    """
    Prepare data for 3 hospitals with different distributions
    """
    print("🏥 Preparing hospital data partitions...")
    
    # Download and load data
    data_path = download_mednist()
    train_loader, val_loader, test_loader, class_names = load_mednist_dataset(data_path)
    
    # Get the train dataset
    train_dataset = train_loader.dataset
    
    # Create non-IID partitions
    client_datasets = create_non_iid_partition(train_dataset, num_clients=3, alpha=0.5)
    
    # Create data loaders for each hospital
    hospital_loaders = []
    for i, client_dataset in enumerate(client_datasets):
        loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        hospital_loaders.append(loader)
        
        # Print distribution for each hospital
        labels = [train_dataset.dataset.targets[idx] for idx in client_dataset.indices]
        distribution = Counter(labels)
        print(f"\n🏥 Hospital {chr(65+i)} (Client {i+1}):")
        print(f"   Total samples: {len(client_dataset)}")
        for class_id, count in distribution.items():
            print(f"   {class_names[class_id]}: {count} ({count/len(client_dataset)*100:.1f}%)")
    
    return hospital_loaders, val_loader, test_loader, class_names

if __name__ == "__main__":
    hospital_loaders, val_loader, test_loader, class_names = prepare_hospital_data()
    print("\n✅ Hospital data preparation complete!")