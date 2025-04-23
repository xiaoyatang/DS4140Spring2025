import numpy as np
import pandas as pd

# Load M (assumed column-stochastic from your earlier example)
M = pd.read_csv("M.csv", header=None).values
# M = M.T  # Convert to row-stochastic for standard interpretation

n = M.shape[0]
q = np.zeros(n)
q[0] = 1

beta = 0.15

# Create teleportation matrix
U = np.ones((n, n)) / n

# Adjusted transition matrix
M_prime = (1 - beta) * M + beta * U

## Eigenvalue
from numpy.linalg import eig
eigvals, eigvecs = eig(M_prime)


# Extract the corresponding eigenvector
v = np.real(eigvecs[:, 0])  # Ensure it's real

# Normalize to make it a probability distribution (L1 normalization)
q_star = v / np.sum(v)


print("q* with teleportation (beta = 0.15):")
print(q_star)
