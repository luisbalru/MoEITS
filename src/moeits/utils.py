import numpy as np
from scipy.stats import entropy
import torch


def compute_information_measures(A, B, bins='auto'):
    """
    I(A;B) Mutual information
    High I(A;B): Indicates a strong relationship between the information contained in matrices A and B.
    Low I(A;B): Suggests little or no shared information between the matrices.
    High Entropy (within a single matrix): Implies a high degree of randomness or uncertainty within that matrix.
    """
    # Flatten matrices
    A_flat = A.flatten()
    B_flat = B.flatten()

    # Discretize into bins
    A_hist, A_edges = np.histogram(A_flat, bins=bins, density=True)
    B_hist, B_edges = np.histogram(B_flat, bins=bins, density=True)
    AB_hist, _, _ = np.histogram2d(A_flat, B_flat, bins=[A_edges, B_edges], density=True)
    #AB_hist, _, _ = np.histogram2d(A_flat, B_flat, bins=bins, density=True)

    # Normalize histograms to probabilities
    P_A = A_hist / np.sum(A_hist)
    P_B = B_hist / np.sum(B_hist)
    P_AB = AB_hist / np.sum(AB_hist)

    # Entropy calculations
    H_A = entropy(P_A)  # H(A)
    H_B = entropy(P_B)  # H(B)
    H_AB = entropy(P_AB.flatten())  # Joint entropy H(A, B)

    # Mutual information
    I_AB = H_A + H_B - H_AB  # I(A; B)

    # Conditional entropy
    H_A_given_B = H_AB - H_B  # H(A|B)
    H_B_given_A = H_AB - H_A  # H(B|A)
    NMI = I_AB/np.sqrt(H_A*H_B) # Normalized mutual information

    return {
        'H(A)': H_A,
        'H(B)': H_B,
        'H(A,B)': H_AB,
        'I(A;B)': I_AB,
        'H(A|B)': H_A_given_B,
        'H(B|A)': H_B_given_A,
        'NMI': NMI
    }



import torch
from tqdm import tqdm

def _estimate_auto_bins(x):
    """Replicates NumPy's 'auto' bin selection logic on CUDA with overflow protection."""
    n = x.numel()
    if n < 2:
        return 1

    # 1. Freedman-Diaconis estimator
    q75, q25 = torch.quantile(x.float(), torch.tensor([0.75, 0.25], device=x.device))
    iqr = q75 - q25
    
    # 2. Sturges estimator
    sturges_bins = int(torch.ceil(torch.log2(torch.tensor(n, dtype=torch.float32)) + 1).item())

    # Add a strict tolerance (1e-6) to prevent tiny IQRs from generating microscopic bin widths
    if iqr > 1e-6:
        fd_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        fd_bins = int(torch.ceil((x.max() - x.min()) / fd_width).item())
        
        # SAFETY CAP: Never allow more bins than there are data points 
        # (You can also hardcode this to something like 100000 if your arrays are massive)
        fd_bins = min(fd_bins, n)
    else:
        fd_bins = 1

    return max(fd_bins, sturges_bins, 1)


def compute_pairwise_nmi_matrix(tensor_3d, bins='auto'):
    """
    Computes a 2D superior triangular matrix of NMI values for all slices in a 3D tensor.
    
    Args:
        tensor_3d: 3D PyTorch array of shape [N, H, W]
        bins: 'auto' or int
        
    Returns:
        NMI_matrix: [N, N] upper triangular torch tensor with 0 diagonal.
    """
    N, H, W = tensor_3d.shape
    device = tensor_3d.device
    
    # Flatten the 2D slices into 1D arrays: Shape [N, H*W]
    X_flat = tensor_3d.view(N, -1)
    total_elements = X_flat.shape[1]
    
    # Preallocate memory. 
    # Storing bins as int32 cuts VRAM usage in half (~17GB instead of ~34GB)
    X_bins = torch.empty((N, total_elements), dtype=torch.int32, device=device)
    num_bins = torch.zeros(N, dtype=torch.long, device=device)
    H_list = torch.zeros(N, dtype=torch.float32, device=device)
    
    print("Step 1/2: Precomputing 1D entropies and bin mapping...")
    for i in tqdm(range(N)):
        x = X_flat[i].float()
        
        b = _estimate_auto_bins(x) if bins == 'auto' else bins
        num_bins[i] = b
        
        min_x, max_x = x.min(), x.max()
        if min_x == max_x: max_x += 1e-6
        
        # Map values to integer bin IDs and store
        x_binned = torch.clamp(((x - min_x) / (max_x - min_x) * b).long(), 0, b - 1)
        X_bins[i] = x_binned.to(torch.int32)
        
        # Compute 1D Histogram
        x_hist = torch.bincount(x_binned, minlength=b).float()
        
        # Filter 0s to speed up entropy calc and prevent NaN
        x_hist = x_hist[x_hist > 0]
        P_x = x_hist / total_elements
        H_list[i] = -torch.sum(P_x * torch.log(P_x))

    print("Step 2/2: Computing Pairwise Joint Entropies & NMI...")
    NMI_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)
    
    # Calculate combinations for the upper triangular matrix
    for i in tqdm(range(N)):
        # Upcast A_bins to int64 only at execution time for torch.bincount compatibility
        A_b = X_bins[i].to(torch.int64) 
        b_A = num_bins[i].item()
        H_A = H_list[i]
        
        for j in range(i + 1, N):
            B_b = X_bins[j].to(torch.int64)
            b_B = num_bins[j].item()
            H_B = H_list[j]
            
            # Asymmetric 2D Histogram map
            AB_bins = A_b * b_B + B_b
            
            # THE FIX: Use unique instead of bincount to avoid memory explosions
            # This implicitly filters out all 0-count bins automatically!
            _, AB_hist = torch.unique(AB_bins, return_counts=True)
            AB_hist = AB_hist.float()
            
            P_AB = AB_hist / total_elements
            H_AB = -torch.sum(P_AB * torch.log(P_AB))
            
            # Calculate Information Metrics
            I_AB = H_A + H_B - H_AB
            denominator = torch.sqrt(H_A * H_B)
            
            # Populate upper triangular matrix securely
            if denominator > 1e-8:
                NMI_matrix[i, j] = I_AB / denominator
            else:
                NMI_matrix[i, j] = 0.0

    return NMI_matrix