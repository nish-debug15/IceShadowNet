import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    """
    Evaluates a trained model on a given dataloader.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.int32)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    # ROC-AUC requires both classes to be present in y_true, handle edge case if batch is small or synthetic
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics['roc_auc'] = float('nan')
        
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    return metrics

def save_confusion_matrix(cm, save_path):
    """Saves the confusion matrix as an image."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Ice", "Potential Ice"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    
if __name__ == '__main__':
    # Dummy test to show harness works
    print("Testing evaluation harness with dummy data...")
    dummy_labels = np.random.randint(0, 2, 100)
    dummy_preds = np.random.randint(0, 2, 100)
    dummy_probs = np.random.rand(100)
    
    metrics = {
        'accuracy': accuracy_score(dummy_labels, dummy_preds),
        'precision': precision_score(dummy_labels, dummy_preds),
        'recall': recall_score(dummy_labels, dummy_preds),
        'f1': f1_score(dummy_labels, dummy_preds),
        'roc_auc': roc_auc_score(dummy_labels, dummy_probs),
        'confusion_matrix': confusion_matrix(dummy_labels, dummy_preds)
    }
    
    print("Dummy Metrics:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v:.4f}")
            
    os.makedirs("reports", exist_ok=True)
    save_confusion_matrix(metrics['confusion_matrix'], "reports/dummy_confusion_matrix.png")
    print("Saved dummy confusion matrix to reports/dummy_confusion_matrix.png")
