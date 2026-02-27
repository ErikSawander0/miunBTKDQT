import torch
import torch.nn as nn
from transformers import AutoImageProcessor, VitPoseForPoseEstimation
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def linear_cka(X, Y):
    """
    Compute Linear CKA between two feature matrices.
    X: (n_samples, n_features_x)
    Y: (n_samples, n_features_y)
    """
    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute CKA
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    
    # Frobenius norm
    hsic_xy = torch.sum(XtX * YtY.T)
    hsic_xx = torch.sum(XtX * XtX)
    hsic_yy = torch.sum(YtY * YtY)
    
    cka = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy) + 1e-8)
    return cka.item()


def extract_features(model, processor, image_paths, device, max_images=500):
    """
    Extract features from all transformer layers for a batch of images.
    Returns dict: {layer_idx: tensor of shape (n_images, n_tokens, hidden_dim)}
    """
    model.eval()
    
    # Storage for each layer
    all_features = {i: [] for i in range(12)}  # 12 layers in ViT-B
    
    for img_path in tqdm(image_paths[:max_images], desc="Extracting features"):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes = [[[0, 0, w, h]]]
        
        inputs = processor(image, boxes=boxes, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True  # This gives us all layer outputs
            )
        
        # hidden_states is a tuple: (embedding, layer1, layer2, ..., layer12)
        # Skip the embedding (index 0), take layers 1-12
        hidden_states = outputs.hidden_states
        
        for i in range(12):
            # Get layer output, flatten spatial dimensions
            layer_feat = hidden_states[i + 1]  # +1 to skip embedding
            layer_feat = layer_feat.squeeze(0)  # Remove batch dim
            layer_feat = layer_feat.mean(dim=0)  # Average over tokens -> (hidden_dim,)
            all_features[i].append(layer_feat.cpu())
    
    # Stack into tensors
    for i in range(12):
        all_features[i] = torch.stack(all_features[i], dim=0)  # type: ignore (n_images, hidden_dim)
    
    return all_features


def compute_cka_matrix(features):
    """
    Compute CKA similarity between all pairs of layers.
    Returns 12x12 similarity matrix.
    """
    n_layers = len(features)
    cka_matrix = np.zeros((n_layers, n_layers))
    
    for i in tqdm(range(n_layers), desc="Computing CKA"):
        for j in range(n_layers):
            if i <= j:
                cka = linear_cka(features[i], features[j])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka  # Symmetric
    
    return cka_matrix


def plot_cka_matrix(cka_matrix, save_path="cka_matrix.png"):
    """
    Plot CKA similarity matrix as heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels([f'L{i+1}' for i in range(12)])
    ax.set_yticklabels([f'L{i+1}' for i in range(12)])
    ax.set_xlabel('Teacher Layer')
    ax.set_ylabel('Teacher Layer')
    ax.set_title('CKA Similarity Between ViTPose-B Layers')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity')
    
    # Add values in cells
    for i in range(12):
        for j in range(12):
            text = ax.text(j, i, f'{cka_matrix[i, j]:.2f}',
                          ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def main():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_name = "usyd-community/vitpose-base-simple"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name)
    model = model.to(device)  # type: ignore
    model.eval()
    
    # Get image paths from COCO val set
    val_dir = "dataset/val2017"
    image_paths = [
        os.path.join(val_dir, f) 
        for f in os.listdir(val_dir) 
        if f.endswith('.jpg')
    ]
    print(f"Found {len(image_paths)} images")
    
    # Extract features (use 500 images for reasonable runtime)
    print("\nExtracting features from teacher...")
    features = extract_features(model, processor, image_paths, device, max_images=500)
    
    # Compute CKA matrix
    print("\nComputing CKA similarity matrix...")
    cka_matrix = compute_cka_matrix(features)
    
    # Save raw matrix
    np.save("cka_matrix.npy", cka_matrix)
    print("Saved raw matrix to cka_matrix.npy")
    
    # Plot
    plot_cka_matrix(cka_matrix)
    
    # Print interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    # Find clusters (layers with >0.9 similarity)
    print("\nHighly similar layer pairs (CKA > 0.9):")
    for i in range(12):
        for j in range(i+1, 12):
            if cka_matrix[i, j] > 0.9:
                print(f"  L{i+1} <-> L{j+1}: {cka_matrix[i, j]:.3f}")
    
    # Average similarity to final layer
    print("\nSimilarity to final layer (L12):")
    for i in range(11):
        print(f"  L{i+1}: {cka_matrix[i, 11]:.3f}")


if __name__ == "__main__":
    main()
