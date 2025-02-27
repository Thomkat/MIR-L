import torch
import numpy as np

def create_mask(model):
    masks = []
    # Collect masks for weight parameters
    for name, param in model.named_parameters():
        if 'weight' in name:
            m = torch.ones_like(param.data).to(param.device)
            masks.append(m)

    weight_mask = masks

    return weight_mask


def prune_and_reset(model, mask, percent, prune_type, initial_state_dict):
    # ======= Prune the network =======

    # Collect weight parameters
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]

    if prune_type == 'global':
        all_weights = np.concatenate([
            param.data.cpu().numpy().flatten()
            for _, param in weight_params
        ])
        
        # Calculate global percentile threshold
        alive = all_weights[np.nonzero(all_weights)]
        global_percentile_value = np.percentile(abs(alive), percent)
    
    step = 0
    for (name, param) in weight_params:
        tensor = param.data.cpu().numpy()
    
        if prune_type == 'global':
            # Use the global threshold if we do global pruning
            threshold = global_percentile_value
        else:
            # The output layer is pruned at half the rate in layer-wise pruning
            if "output" in name:  
                effective_percent = percent / 2.0
            else:
                effective_percent = percent

            # Compute layer-wise threshold
            alive = tensor[np.nonzero(tensor)]
            threshold = np.percentile(abs(alive), effective_percent)

        old_mask = mask[step].cpu().numpy()  # was a GPU tensor, go to NumPy on CPU

        # Create a new mask: 0 where |tensor| < threshold, else old_mask
        new_mask_np = np.where(np.abs(tensor) < threshold, 0, old_mask)

        # Convert to a GPU Tensor
        mask[step] = torch.from_numpy(new_mask_np).to(param.device, non_blocking=True)

        # Apply the new mask to the parameter
        param.data.mul_(mask[step])
        step += 1



    # ======= Reset surviving weights to their initial values =======

    # Now that the mask is updated, reset the weights to their original values,
    # move the mask to the same device as the weights and mask them
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    step = 0
    for (name, param) in weight_params:
        weight_dev = param.device
        original_weight = initial_state_dict[name].to(weight_dev)

        mask[step] = mask[step].to(weight_dev)
        param.data = mask[step] * original_weight
        step += 1

    # Apply original biases
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.data = initial_state_dict[name].to(param.device)

def print_layer_info(model):
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        non_zero_params = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        print(
            f"{name:20} "
            f"|| Non-zero parameters = {non_zero_params} / {total_params} "
            f"|| Pruned parameters = {total_params - non_zero_params} "
            f"|| Compression rate: {total_params / non_zero_params:.2f}x "
            f"({100 * (total_params - non_zero_params) / total_params:.2f}% pruned) "
            f"|| Layer Shape = {tensor.shape}"
        )
