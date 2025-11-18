import numpy as np
from scipy.stats import entropy



def compute_information_measures(A, B, bins='auto'):
    """
    I(A;B) Mutual infonrmation
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