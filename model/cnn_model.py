import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalCNN(nn.Module):
    """
    CNN for medical image classification (Chest X-ray)
    Optimized for 224x224 images
    """
    def __init__(self, num_classes=3):
        super(MedicalCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate flattened size
        self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _calculate_flattened_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
            dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))
            dummy = self.pool(F.relu(self.bn4(self.conv4(dummy))))
            self.flattened_size = dummy.view(1, -1).size(1)
    
    def forward(self, x):
        # Convolution + Pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers with dropout
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# For testing
if __name__ == "__main__":
    model = MedicalCNN(num_classes=3)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")