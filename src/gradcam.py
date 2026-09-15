import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import models to define target layers
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def generate_gradcam_manual(model, target_layer, input_tensor, original_image):
    """
    Manual Grad-CAM implementation in case `grad-cam` package is not available 
    or fails on custom architectures.
    """
    # This is a placeholder for manual implementation. 
    # For now, we rely on the package or generate a dummy heatmap if package missing.
    print("Manual GradCAM fallback triggered.")
    heatmap = np.random.rand(original_image.shape[0], original_image.shape[1])
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

def run_gradcam(model, input_tensor, original_image, model_type="cnn", save_path="reports/gradcam.png"):
    """
    Generates and saves a Grad-CAM overlay.
    """
    model.eval()
    
    # Identify target layer based on model type
    if model_type == "cnn":
        # Target the last convolutional block in CustomCNN
        target_layers = [model.conv3]
    elif model_type == "resnet":
        # Target the last layer of ResNet
        target_layers = [model.model.layer4[-1]]
    else:
        raise ValueError("Unknown model_type. Use 'cnn' or 'resnet'.")

    if HAS_GRADCAM:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)] # Binary classification, index 0
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay on original image (assuming original_image is normalized 0-1 RGB/Grayscale)
        # We'll use just the first channel for visualization if it's SAR
        vis_img = original_image[:, :, 0] if len(original_image.shape) == 3 else original_image
        # Replicate to 3 channels for `show_cam_on_image`
        vis_img = np.stack([vis_img]*3, axis=-1)
        
        visualization = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original (Ch 0)')
        plt.imshow(vis_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Grad-CAM')
        plt.imshow(visualization)
        plt.axis('off')
        
        plt.savefig(save_path)
        plt.close()
        print(f"Grad-CAM saved to {save_path}")
    else:
        print("pytorch-grad-cam not installed. Generating dummy manual overlay.")
        heatmap = generate_gradcam_manual(model, None, input_tensor, original_image)
        plt.imshow(heatmap, cmap='jet')
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    # Dummy execution
    print("Testing Grad-CAM script...")
    dummy_input = torch.randn(1, 2, 256, 256)
    dummy_img = np.random.rand(256, 256, 2).astype(np.float32)
    
    cnn_model = CustomCNN(in_channels=2)
    os.makedirs("reports", exist_ok=True)
    run_gradcam(cnn_model, dummy_input, dummy_img, model_type="cnn", save_path="reports/cnn_gradcam_dummy.png")
    
    resnet_model = IceResNet(in_channels=2, pretrained=False)
    run_gradcam(resnet_model, dummy_input, dummy_img, model_type="resnet", save_path="reports/resnet_gradcam_dummy.png")
