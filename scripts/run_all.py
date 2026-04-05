import subprocess
import threading
import time
import sys
import os

def run_server():
    """Run the FL server"""
    print("🚀 Starting FL Server...")
    subprocess.run([sys.executable, "server/server.py"])

def run_client(client_name):
    """Run a client/hospital"""
    print(f"🏥 Starting {client_name}...")
    subprocess.run([sys.executable, f"clients/{client_name.lower()}.py"])

def main():
    print("="*60)
    print("🌟 FEDERATED MEDICAL IMAGING SYSTEM")
    print("="*60)
    print("This will launch:")
    print("  - 1 Central Server")
    print("  - 3 Hospitals (A, B, C)")
    print("="*60)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("\n⏳ Waiting for server to start...")
    time.sleep(5)
    
    # Start all clients
    clients = ["hospital_a", "hospital_b", "hospital_c"]
    client_threads = []
    
    for client in clients:
        thread = threading.Thread(target=run_client, args=(client,))
        thread.start()
        client_threads.append(thread)
        time.sleep(2)  # Stagger client starts
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Federated Learning System...")

if __name__ == "__main__":
    main()