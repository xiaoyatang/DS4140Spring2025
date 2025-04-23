import pandas as pd
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

A = pd.read_csv('A.csv')

# Assume the first column contains labels (e.g., words)
labels = A.iloc[:, 0]        # This is a pandas Series of labels
A = A.iloc[:, 1:].to_numpy(dtype=float)

# Compute SVD
U, s, Vt = LA.svd(A, full_matrices=False)

# Take the top 2 right singular vectors
Vt2 = Vt[:2, :]  # shape (2, n_features)

# Project A onto top 2 right singular vectors
A_proj_2d = A @ Vt2.T  # shape (n_samples, 2)

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(A_proj_2d[:, 0], A_proj_2d[:, 1], alpha=0.7)

# Optionally annotate some points with their labels
for i in range(min(30, len(labels))):  # Show top 30 for readability
    plt.text(A_proj_2d[i, 0], A_proj_2d[i, 1], str(labels[i]), fontsize=8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection of A onto Top 2 Right Singular Vectors')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
