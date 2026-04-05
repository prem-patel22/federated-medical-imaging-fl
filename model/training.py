import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class ModelTrainer:
    """Handles local training and evaluation for each hospital"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_one_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': running_loss/(batch_idx+1),
                'Acc': 100.*correct/total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_local(self, train_loader, local_epochs=5, lr=0.001):
        """Complete local training for multiple epochs"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"🏥 Local training on {self.device} for {local_epochs} epochs...")
        
        for epoch in range(1, local_epochs + 1):
            loss, acc = self.train_one_epoch(train_loader, optimizer, epoch)
            print(f"   Epoch {epoch}/{local_epochs} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
        
        return self.get_model_weights()
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_model_weights(self):
        """Extract model weights for sending to server"""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights):
        """Set model weights received from server"""
        with torch.no_grad():
            for k, v in weights.items():
                self.model.state_dict()[k].copy_(torch.from_numpy(v))