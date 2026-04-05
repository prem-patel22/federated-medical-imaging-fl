import json
import os
from datetime import datetime
import pandas as pd

class MetricsLogger:
    """
    Captures real training metrics from Federated Learning
    Saves to JSON file for dashboard to read
    """
    
    def __init__(self, log_file="training_metrics.json"):
        self.log_file = log_file
        self.metrics = {
            "rounds": [],
            "hospital_a": {"accuracies": [], "losses": [], "samples": 930},
            "hospital_b": {"accuracies": [], "losses": [], "samples": 1044},
            "hospital_c": {"accuracies": [], "losses": [], "samples": 497},
            "global": {"accuracies": [], "losses": []},
            "privacy": {"epsilons": []},
            "timestamp": []
        }
        self.load_existing()
    
    def load_existing(self):
        """Load existing metrics if file exists"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = data
                print(f"📊 Loaded {len(self.metrics['rounds'])} rounds of training data")
            except:
                print("📊 Starting fresh metrics log")
    
    def save(self):
        """Save metrics to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Metrics saved to {self.log_file}")
    
    def log_round_start(self, round_num):
        """Log start of a training round"""
        self.metrics["rounds"].append(round_num)
        self.metrics["timestamp"].append(datetime.now().isoformat())
        self.save()
        print(f"🔄 Round {round_num} started at {datetime.now()}")
    
    def log_hospital_accuracy(self, round_num, hospital_name, accuracy, loss):
        """Log individual hospital performance after training"""
        hospital_key = hospital_name.lower().replace(" ", "_")
        if hospital_key in self.metrics:
            self.metrics[hospital_key]["accuracies"].append({
                "round": round_num,
                "value": accuracy
            })
            self.metrics[hospital_key]["losses"].append({
                "round": round_num,
                "value": loss
            })
        self.save()
        print(f"📊 {hospital_name} - Round {round_num}: {accuracy:.2f}% accuracy")
    
    def log_global_metrics(self, round_num, accuracy, loss):
        """Log global model metrics after aggregation"""
        self.metrics["global"]["accuracies"].append({
            "round": round_num,
            "value": accuracy
        })
        self.metrics["global"]["losses"].append({
            "round": round_num,
            "value": loss
        })
        self.save()
        print(f"🌍 Global Model - Round {round_num}: {accuracy:.2f}% accuracy")
    
    def log_privacy_budget(self, round_num, epsilon):
        """Log privacy budget consumption"""
        self.metrics["privacy"]["epsilons"].append({
            "round": round_num,
            "value": epsilon
        })
        self.save()
        print(f"🔒 Privacy Budget - Round {round_num}: ε={epsilon:.2f}")
    
    def get_realtime_data(self):
        """Get latest metrics for dashboard"""
        return self.metrics

# Create global logger instance
logger = MetricsLogger()