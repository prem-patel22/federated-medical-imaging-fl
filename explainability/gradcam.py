import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class GradCAM:
    """
    Grad-CAM: Visualize which parts of the image influenced the model's decision
    Perfect for doctors to trust the AI!
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activations from forward pass"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients from backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for an image
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax().item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU activation
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
    
    def visualize(self, image_tensor, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image
        """
        # Convert tensor to numpy
        if isinstance(image_tensor, torch.Tensor):
            image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
        else:
            image = image_tensor
        
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on image
        overlay = (1 - alpha) * image + alpha * (heatmap / 255.0)
        overlay = np.clip(overlay, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (Why AI Made Decision)")
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig

# Demo function
def demo_gradcam():
    """
    Quick demo of Grad-CAM
    """
    print("🎨 Grad-CAM Demo")
    print("="*50)
    
    from model.cnn_model import MedicalCNN
    
    # Load model
    model = MedicalCNN(num_classes=3)
    
    # Get the last convolutional layer
    target_layer = model.conv4
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Create dummy image
    dummy_image = torch.randn(3, 224, 224)
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(dummy_image, target_class=0)
    
    # Visualize
    fig = gradcam.visualize(dummy_image, heatmap)
    
    print("✅ Grad-CAM ready! Shows doctors why model makes predictions")
    return fig

if __name__ == "__main__":
    fig = demo_gradcam()
    plt.show()