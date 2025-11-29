import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt

# --- 1. The DeepFool Algorithm ---
def deepfool(model, image, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool Algorithm Implementation.
    
    Args:
        model: The PyTorch model (must be in eval mode).
        image: Input tensor of shape (1, 5, 64, Time).
        num_classes: Number of classes (10 for UrbanSound8K).
        overshoot: A tiny value to push the example PAST the boundary.
        max_iter: Maximum number of loops to try before giving up.
    
    Returns:
        r_tot: The total noise added (Tensor).
        perturbed_image: The final adversarial example (Tensor).
        success: Boolean (True if the model changed its prediction).
    """
    device = image.device
    
    # Clone image to track gradients
    f_image = image.clone().detach().requires_grad_(True)
    
    # Get initial prediction
    output = model(f_image)
    _, initial_pred = torch.max(output, 1)
    initial_pred = initial_pred.item()
    
    current_pred = initial_pred
    perturbed_image = image.clone().detach()
    input_shape = image.shape
    
    # Accumulators for noise
    r_tot = torch.zeros(input_shape).to(device)

    loop_i = 0

    while (current_pred == initial_pred) and (loop_i < max_iter):
        
        # Start new gradient tracking for this iteration
        x = perturbed_image.clone().detach().requires_grad_(True)
        fs = model(x)
        
        # Backpropagate the winning class score
        fs[0, initial_pred].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        pert_min = np.inf
        w_best = None
        
        # Check distance to all other classes
        for k in range(num_classes):
            if k == initial_pred: continue
                
            x.grad.zero_()
            fs[0, k].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()
            
            # Geometry of the boundary
            w_k = cur_grad - grad_orig
            f_k = (fs[0, k] - fs[0, initial_pred]).data
            
            # Calculate distance (norm of 5-channel flattened vector)
            pert = abs(f_k) / torch.norm(w_k.flatten())
            
            if pert < pert_min:
                pert_min = pert
                w_best = w_k

        # Calculate perturbation vector r_i
        r_i = (pert_min + 1e-4) * w_best / torch.norm(w_best)
        r_tot = r_tot + r_i

        # Apply noise to original image
        perturbed_image = image + (1 + overshoot) * r_tot
        
        # Check if fooled
        output = model(perturbed_image) 
        _, current_pred = torch.max(output, 1)
        current_pred = current_pred.item()
        
        loop_i += 1

    success = (current_pred != initial_pred)
    return r_tot, perturbed_image, success


# --- 2. Testing & Visualization Helpers ---

def analyze_perturbation(perturbation_tensor):
    """Calculates which of the 5 channels received the most noise."""
    noise_numpy = perturbation_tensor.squeeze(0).cpu().detach().numpy()
    # Sum of squared error per channel
    channel_energy = np.sqrt(np.sum(noise_numpy**2, axis=(1, 2)))
    channels = ['Log Mel', 'Delta', 'Delta-Delta', 'Harmonic', 'Percussive']
    
    print("   [Noise Analysis] Energy per channel:")
    for i, name in enumerate(channels):
        print(f"     - {name}: {channel_energy[i]:.4f}")
    
    max_idx = np.argmax(channel_energy)
    return channels[max_idx]

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Import your custom modules
    try:
        from dataloader import Dataloader
        from preprocessing import AudioPreprocessor
    except ImportError:
        print("Error: Could not import Dataloader or Preprocessing. Make sure files are in the same folder.")
        exit()

    # --- A. Setup a Dummy Model for Testing ---
    # (In your real project, load your trained .pth file here instead)
    class SimpleTestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: 5 channels, Output: 10 classes
            self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Forces output to (Batch, 32, 1, 1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)

    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    model = SimpleTestCNN().to(device)
    model.eval() # CRITICAL: DeepFool fails if model is in train mode (Dropout/BatchNorm)

    # --- B. Initialize Data Pipeline ---
    # UPDATE THIS PATH to your actual datasets folder
    MY_DATASET_PATH = r"C:\deep-learning-urban-sound-data\datasets"
    
    try:
        prep = AudioPreprocessor()
        dl = Dataloader(dataset_path=MY_DATASET_PATH, preprocessing=prep, verbose=False)
        print(f"Data loaded. Total clips: {len(dl)}\n")
    except Exception as e:
        print(f"Could not load dataset at {MY_DATASET_PATH}. Error: {e}")
        exit()

    # --- C. Test on a Couple of Examples ---
    test_indices = [10, 42, 100] # Let's test 3 random clips
    
    for idx in test_indices:
        print("="*60)
        print(f"Testing Clip Index: {idx}")
        
        # 1. Get Data
        features, fold, label_idx = dl[idx]
        original_label = dl.get_label_mapping()[label_idx]
        
        # 2. Prepare Tensor (Add Batch Dim: 5,64,T -> 1,5,64,T)
        input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)
        
        # 3. Run DeepFool
        print(f"Original Label: {original_label}")
        print("Running attack...")
        
        noise, adv_img, success = deepfool(model, input_tensor, max_iter=20)
        
        if success:
            # Check new prediction
            with torch.no_grad():
                new_out = model(adv_img)
                new_pred = torch.argmax(new_out, dim=1).item()
                new_label = dl.get_label_mapping()[new_pred]
            
            print(f"SUCCESS! Model fooled: {original_label} -> {new_label}")
            
            # Analyze which channel took the most damage
            worst_channel = analyze_perturbation(noise)
            print(f"   Most attacked channel: {worst_channel}")
            
            # 4. Visualizations
            print("Displaying plots...")
            
            # Convert tensors back to numpy for your plotting function
            adv_numpy = adv_img.squeeze(0).cpu().detach().numpy()
            noise_numpy = noise.squeeze(0).cpu().detach().numpy()
            
            # Plot 1: The Original (Reference)
            # dl.preprocessing.plot_all_channels(features, fold, f"Original: {original_label}")
            
            # Plot 2: The Adversarial (Result)
            dl.preprocessing.plot_all_channels(adv_numpy, fold, f"Adversarial: {new_label}")
            
            # Plot 3: The Noise (What DeepFool added)
            # We multiply by 100 because the noise is usually invisible to the eye
            dl.preprocessing.plot_all_channels(noise_numpy * 100, fold, f"Pure Noise (x100) - Target: {new_label}")
            
        else:
            print("Failed to fool the model (Example might be too robust or max_iter too low).")