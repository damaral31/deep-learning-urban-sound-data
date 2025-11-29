import torch
import numpy as np
import copy

def deepfool(model, image, num_classes=10, overshoot=0.02, max_iter=50):
    """
    Args:
        model: The PyTorch model (in eval mode).
        image: Input tensor of shape (1, 5, 64, Time).
        num_classes: 10 for UrbanSound8K.
        overshoot: Small value to push the input just past the boundary.
        max_iter: Maximum loops to try before giving up.
    
    Returns:
        minimal_perturbation: The noise added (Tensor).
        perturbed_image: The adversarial example (Tensor).
        success: Boolean, true if the model was fooled.
    """
    device = image.device
    f_image = image.clone().detach().requires_grad_(True)
    
    # 1. Forward pass to get initial prediction
    output = model(f_image)
    _, initial_pred = torch.max(output, 1)
    initial_pred = initial_pred.item()
    
    current_pred = initial_pred
    perturbed_image = image.clone().detach()
    input_shape = image.shape
    
    # Variable to accumulate the total noise
    w_tot = torch.zeros(input_shape).to(device)
    r_tot = torch.zeros(input_shape).to(device)

    loop_i = 0
    while (current_pred == initial_pred) and (loop_i < max_iter):
        
        # Start a new gradient calculation for this iteration
        x = perturbed_image.clone().detach().requires_grad_(True)
        fs = model(x)
        
        # Backpropagate the score of the predicted class (k_i)
        fs[0, initial_pred].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        pert_min = np.inf
        w_best = None
        
        # 2. Check distance to all other 9 classes to find the closest boundary
        for k in range(num_classes):
            if k == initial_pred:
                continue
                
            # Zero gradients before next backward pass
            x.grad.zero_()
            fs[0, k].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()
            
            # Calculate vector w_k and f_k (Geometry of the boundary)
            w_k = cur_grad - grad_orig
            f_k = (fs[0, k] - fs[0, initial_pred]).data
            
            # Calculate the magnitude of the perturbation needed for this class
            pert = abs(f_k) / torch.norm(w_k.flatten())
            
            # If this class boundary is closer than previous ones, save it
            if pert < pert_min:
                pert_min = pert
                w_best = w_k

        # 3. Calculate the perturbation vector r_i
        # r_i = (pert_min + overshoot) * (w_best / norm(w_best))
        r_i = (pert_min + 1e-4) * w_best / torch.norm(w_best)
        r_tot = r_tot + r_i

        # 4. Apply perturbation to the image
        perturbed_image = image + (1 + overshoot) * r_tot
        
        # Re-check prediction
        output = model(perturbed_image) 
        _, current_pred = torch.max(output, 1)
        current_pred = current_pred.item()
        
        loop_i += 1

    success = (current_pred != initial_pred)
    return r_tot, perturbed_image, success