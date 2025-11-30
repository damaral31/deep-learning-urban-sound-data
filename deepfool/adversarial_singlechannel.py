import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
import librosa.display

# --- 1. The DeepFool Algorithm ---
def deepfool(model, image, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool Algorithm Implementation for Single Channel.
    
    Args:
        model: The PyTorch model (must be in eval mode).
        image: Input tensor of shape (1, 1, 64, Time).
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
            
            # Calculate distance
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
    return r_tot, perturbed_image, success, loop_i


# --- 2. Testing & Visualization Helpers ---
def analyze_perturbation(perturbation_tensor):
    """Calculates perturbation energy for singlechannel model."""
    noise_numpy = perturbation_tensor.squeeze(0).cpu().detach().numpy()
    total_energy = np.sqrt(np.sum(noise_numpy**2))
    print(f"   [Noise Analysis] Total perturbation energy: {total_energy:.4f}")
    return total_energy

def calculate_relative_robustness(original_input, perturbation):
    """
    Calculate Relative Robustness: ||r|| / ||x||
    Where r is the perturbation and x is the original input.
    Lower values indicate less robust models (easier to fool).
    """
    original_norm = torch.norm(original_input.flatten()).item()
    perturbation_norm = torch.norm(perturbation.flatten()).item()
    relative_robustness = perturbation_norm / original_norm
    return relative_robustness, perturbation_norm, original_norm

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Import your custom modules
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
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
    model.eval()
    print(f"Singlechannel model loaded from: {model_path}\n")

    # Initialize Data Pipeline
    MY_DATASET_PATH = r"C:\deep-learning-urban-sound-data\datasets"
    
    try:
        prep = AudioPreprocessor()
        dl = Dataloader(dataset_path=MY_DATASET_PATH, preprocessing=prep, verbose=False, 
                       folds=["fold9", "fold10"], use_cache=True, include_augmented=False)
        print(f"Data loaded. Total clips from folds 9 and 10: {len(dl)}\n")
    except Exception as e:
        print(f"Could not load dataset at {MY_DATASET_PATH}. Error: {e}")
        exit()

    # Test on all samples from folds 9 and 10
    import csv
    import os
    
    # Prepare CSV output
    output_dir = r"C:\deep-learning-urban-sound-data\deepfool\results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "singlechannel_adversarial_results.csv")
    
    # Get all indices (all samples are from folds 9 and 10)
    test_indices = list(range(len(dl)))
    
    print(f"Testing on {len(test_indices)} samples\n")
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['sample_idx', 'fold', 'real_class', 'initial_pred', 'initial_correct', 'fooled_class', 'relative_robustness', 'perturbation_norm', 'original_norm', 'num_iterations', 'success'])
        
        for idx in test_indices:
            print("="*60)
            print(f"Testing Clip Index: {idx}")
            
            # Get Data
            features, fold, label_idx = dl[idx]
            original_label = dl.get_label_mapping()[label_idx]
            
            # Prepare Tensor - Select first channel only (Log-Mel Spectrogram)
            # features shape: (5, 64, T) -> select first channel -> (1, 64, T) -> add batch dim -> (1, 1, 64, T)
            input_tensor = torch.from_numpy(features[0:1]).unsqueeze(0).float().to(device)
            
            # Check initial prediction
            with torch.no_grad():
                initial_output = model(input_tensor)
                initial_pred_idx = torch.argmax(initial_output, dim=1).item()
                initial_pred_label = dl.get_label_mapping()[initial_pred_idx]
                initial_correct = (initial_pred_idx == label_idx)
            
            print(f"Real Label: {original_label}")
            print(f"Initial Prediction: {initial_pred_label} {'✓' if initial_correct else '✗'}")
            
            if not initial_correct:
                print("⚠ Model already misclassifies this sample. Skipping attack.")
                csv_writer.writerow([idx, fold, original_label, initial_pred_label, False, initial_pred_label, 0.0, 0.0, 0.0, 0, False])
                csvfile.flush()
                continue
            
            # Run DeepFool
            print("Running attack...")
            
            noise, adv_img, success, num_iters = deepfool(model, input_tensor, max_iter=50)
            
            fooled_label = original_label
            rel_robustness = 0.0
            pert_norm = 0.0
            orig_norm = 0.0
            
            if success:
                # Check new prediction
                with torch.no_grad():
                    new_out = model(adv_img)
                    new_pred = torch.argmax(new_out, dim=1).item()
                    fooled_label = dl.get_label_mapping()[new_pred]
                
                print(f"SUCCESS! Model fooled: {original_label} -> {fooled_label}")
                
                # Calculate Relative Robustness
                rel_robustness, pert_norm, orig_norm = calculate_relative_robustness(input_tensor, noise)
                print(f"   [Robustness Analysis]")
                print(f"     - Original input norm: {orig_norm:.4f}")
                print(f"     - Perturbation norm: {pert_norm:.4f}")
                print(f"     - Relative Robustness (||r||/||x||): {rel_robustness:.6f}")
                print(f"     - Iterations needed: {num_iters}")
            else:
                print("Failed to fool the model")
                # Still calculate norms for failed attempts
                rel_robustness, pert_norm, orig_norm = calculate_relative_robustness(input_tensor, noise)
            
            # Write to CSV
            csv_writer.writerow([idx, fold, original_label, initial_pred_label, initial_correct, fooled_label, rel_robustness, pert_norm, orig_norm, num_iters, success])
            csvfile.flush()  # Ensure data is written immediately
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_path}")
    print(f"Total samples tested: {len(test_indices)}")
