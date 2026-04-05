import flwr as fl
from flwr.server.strategy import FedAvg
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics_logger import logger

class CustomStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_round = 0
    
    def aggregate_fit(self, rnd, results, failures):
        # Log round start
        self.current_round = rnd
        logger.log_round_start(rnd)
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Calculate global accuracy (average of hospitals)
        if results:
            accuracies = [metrics.get("accuracy", 0) for _, metrics in results]
            global_accuracy = sum(accuracies) / len(accuracies)
            logger.log_global_metrics(rnd, global_accuracy, 0)
        
        return aggregated_parameters, aggregated_metrics

print("="*60)
print("🏥 Federated Learning Server Starting...")
print("="*60)

# Create strategy with logging
strategy = CustomStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10)
)