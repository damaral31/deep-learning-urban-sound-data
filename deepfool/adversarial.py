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

def analyze_perturbation(perturbation_tensor, num_channels=1):
    """Calculates perturbation energy for singlechannel or multichannel models."""
    noise_numpy = perturbation_tensor.squeeze(0).cpu().detach().numpy()
    
    if num_channels == 1:
        # For singlechannel: just calculate total energy
        total_energy = np.sqrt(np.sum(noise_numpy**2))
        print(f"   [Noise Analysis] Total perturbation energy: {total_energy:.4f}")
        return "Log Mel Spectrogram"
    else:
        # For multichannel: analyze per channel
        channel_energy = np.sqrt(np.sum(noise_numpy**2, axis=(1, 2)))
        channels = ['Log Mel', 'Delta', 'Delta-Delta', 'Harmonic', 'Percussive']
        
        print("   [Noise Analysis] Energy per channel:")
        for i, name in enumerate(channels[:num_channels]):
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

    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Load the trained singlechannel CNN model
    from models.CNN import SoundCNN
    model = SoundCNN(num_classes=10, SqueezeExcitation=False, in_channels=1)
    model_path = r"C:\deep-learning-urban-sound-data\models_for_adversarial\CNN_singlechannel_all_folds.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: DeepFool fails if model is in train mode (Dropout/BatchNorm)
    print(f"Model loaded from: {model_path}")

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
        
        # 2. Prepare Tensor (Add Batch Dim and select first channel for singlechannel model)
        # features shape: (5, 64, T) -> select first channel -> (1, 64, T) -> add batch dim -> (1, 1, 64, T)
        input_tensor = torch.from_numpy(features[0:1]).unsqueeze(0).float().to(device)
        
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
            
            # Analyze perturbation
            worst_channel = analyze_perturbation(noise, num_channels=1)
            print(f"   Channel used: {worst_channel}")
            
            # 4. Visualizations
            print("Displaying plots...")
            
            # Convert tensors back to numpy for plotting
            # For singlechannel: shape is (1, 1, 64, T) -> extract to (1, 64, T) for plotting
            adv_numpy = adv_img.squeeze(0).cpu().detach().numpy()
            noise_numpy = noise.squeeze(0).cpu().detach().numpy()
            
            # Create single-channel plot using matplotlib
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Plot 1: Original
            import librosa.display
            librosa.display.specshow(features[0], x_axis='time', y_axis='mel', 
                                    sr=prep.sample_rate, ax=axes[0])
            axes[0].set_title(f"Original: {original_label}")
            fig.colorbar(axes[0].images[0], ax=axes[0], format='%+2.0f dB')
            
            # Plot 2: Adversarial
            librosa.display.specshow(adv_numpy[0], x_axis='time', y_axis='mel', 
                                    sr=prep.sample_rate, ax=axes[1])
            axes[1].set_title(f"Adversarial: {new_label}")
            fig.colorbar(axes[1].images[0], ax=axes[1], format='%+2.0f dB')
            
            # Plot 3: Noise (magnified)
            librosa.display.specshow(noise_numpy[0] * 100, x_axis='time', y_axis='mel', 
                                    sr=prep.sample_rate, ax=axes[2])
            axes[2].set_title(f"Pure Noise (x100) - Target: {new_label}")
            fig.colorbar(axes[2].images[0], ax=axes[2], format='%+2.0f dB')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("Failed to fool the model (Example might be too robust or max_iter too low).")