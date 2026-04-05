import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn_model import MedicalCNN
from data.medical_dataset import prepare_hospital_data
from utils.metrics_logger import logger

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, val_loader, round_num=0):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.round_num = round_num
        self.model = MedicalCNN(num_classes=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        print(f"✅ Hospital {client_id}: {len(train_loader.dataset)} training samples on {self.device}")
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def fit(self, parameters, config):
        # Get round number from config
        self.round_num = config.get("server_round", 0)
        
        # Set model weights
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Local training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        epoch_accuracies = []
        epoch_losses = []
        
        for epoch in range(3):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in self.train_loader:
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
            
            epoch_acc = 100. * correct / total
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracies.append(epoch_acc)
            epoch_losses.append(epoch_loss)
            print(f"   Hospital {self.client_id} - Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        # Log final accuracy for this round
        final_accuracy = epoch_accuracies[-1]
        final_loss = epoch_losses[-1]
        
        # LOG REAL DATA!
        logger.log_hospital_accuracy(
            self.round_num, 
            f"Hospital {self.client_id}", 
            final_accuracy, 
            final_loss
        )
        
        print(f"🏥 Hospital {self.client_id} completed round {self.round_num}")
        return self.get_parameters(config), len(self.train_loader.dataset), {"accuracy": final_accuracy}
    
    def evaluate(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.val_loader)
        print(f"📊 Hospital {self.client_id} - Validation Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return float(avg_loss), len(self.val_loader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"\n🏥 Starting Hospital B...")
    hospital_loaders, val_loader, test_loader, class_names = prepare_hospital_data()
    
    # Hospital B uses index 1 (second loader)
    client = HospitalClient("B", hospital_loaders[1], val_loader)
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )