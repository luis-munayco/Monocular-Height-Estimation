from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def calculate_weights(train_loader: DataLoader, val_loader: DataLoader, bin_width=1, device='cpu'):
    # 3. Collect all height values from both train_loader and val_loader
    all_height_values = []

    # Collect from train_loader
    for batch in train_loader:
        for mask in batch['mask']:
            all_height_values.append(mask.flatten())

    # Collect from val_loader
    for batch in val_loader:
        for mask in batch['mask']:
            all_height_values.append(mask.flatten())

    # Convert the list of tensors to a single PyTorch tensor
    all_height_values_tensor = torch.cat(all_height_values, dim=0)
    all_height_values_tensor = all_height_values_tensor.to(device, dtype=torch.float32)

    # Create the histogram
    hist_tensor = torch.histc(all_height_values_tensor, bins=int(all_height_values_tensor.max()) - int(all_height_values_tensor.min()) + 1,
                              min=int(all_height_values_tensor.min()),max=int(all_height_values_tensor.max())+1)

    # Calculate the inverse frequency
    inverse_hist_tensor = torch.zeros_like(hist_tensor, dtype=torch.float32)
    non_zero_indices = hist_tensor != 0
    inverse_hist_tensor[non_zero_indices] = 1 / torch.pow(hist_tensor[non_zero_indices],0.2)

    # Normalize the inverse frequency
    sum_inverse_hist = torch.sum(inverse_hist_tensor)
    normalized_hist_tensor = inverse_hist_tensor / sum_inverse_hist

    # Ensure normalized_hist_tensor sum is 1
    assert torch.isclose(torch.sum(normalized_hist_tensor), torch.tensor(1.0)), "Normalization failed: Sum of normalized histogram is not 1"

     # 4. Calculate and plot histogram with 1-meter bins
    # Calculate bin edges for the histogram
    min_value = int(all_height_values_tensor.min())
    max_value = int(all_height_values_tensor.max())
    bins = np.arange(min_value, max_value + bin_width, bin_width)
    # Convert tensors to NumPy arrays
    hist_np = hist_tensor.cpu().numpy()
    inverse_hist_np = inverse_hist_tensor.cpu().numpy()
    normalized_hist_np = normalized_hist_tensor.cpu().numpy()

    # Create a DataFrame for histogram and weights
    data = {'Bin_Edges': bins, 'Histogram':hist_np, 'Weights': inverse_hist_np,'Normalized_Weights': normalized_hist_np}
    df_weights = pd.DataFrame(data)
    df_weights.drop(df_weights[df_weights.Weights==0].index, inplace=True)
    df_weights.to_csv('weights.csv')
    
    return df_weights

def weights_tensor(heights_tensor, weights_df,field):
    # Flatten the tensor and concatenate batches
    flattened_heights = heights_tensor.view(-1)

    # Get the bin edges and weights from the DataFrame
    bin_edges = torch.tensor(weights_df['Bin_Edges'].values,dtype=torch.float32, device=heights_tensor.device)
    weights = torch.tensor(weights_df[field].values,dtype=torch.float32, device=heights_tensor.device)

    # Identify the corresponding weights for each height value
    indices = torch.bucketize(flattened_heights, bin_edges)
    weights_tensor = weights[indices]

    # Reshape the tensor back to its original shape
    weights_tensor=weights_tensor.view(heights_tensor.shape)

    return weights_tensor


