import numpy as np
from scipy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

A = pd.read_csv('A.csv')

# Assume the first column contains labels (e.g., words)
labels = A.iloc[:, 0]        # This is a pandas Series of labels
A = A.iloc[:, 1:].to_numpy(dtype=float)

# Step 1: Center the data
mean_vector = np.mean(A, axis=0)
A_centered = A - mean_vector

# Step 2: Compute SVD on centered data
U, s, Vt = LA.svd(A_centered, full_matrices=False)

# Step 3: Find norm of the centered matrix
norm_A_centered = LA.norm(A_centered, 2)
print("L2 Norm of A:", norm_A_centered)
threshold = 0.05 * norm_A_centered
print('threshold:', threshold)
# # Step 4: Loop over k to find the smallest k that gives error < threshold
# for k in range(1, len(s) + 1):
#     Uk = U[:, :k]
#     Sk = np.diag(s[:k])
#     Vtk = Vt[:k, :]
#     Ak = Uk @ Sk @ Vtk
#     error = LA.norm(A_centered - Ak, 2)
    
#     if error < threshold:
#         print(f"Smallest k such that ||A~ - A_k|| < 5% of ||A~||: k = {k}")
#         print(f"||A~||_2 = {norm_A_centered:.4f}")
#         print(f"||A~ - A_k||_2 = {error:.4f}")
#         break
s = np.diag(s)
Ak = list()
l2_norms = []
for k in range(1,51):
    Ak_k = U[:,:k] @ s[:k,:k]@Vt[:k,:]
    Ak.append(Ak_k)
    norm_diff = LA.norm(A_centered - Ak_k, 2)
    l2_norms.append(norm_diff)
    if norm_diff < threshold:
        print(f"Smallest k such that ||A~ - A_k|| < 5% of ||A~||: k = {k}")
        print(f"||A~||_2 = {norm_A_centered:.4f}")
        print(f"||A~ - A_k||_2 = {norm_diff:.4f}")
        break

# print("k | L2 Norm")
# for k, norm in enumerate(l2_norms, start=1):
#     print(f"{k} | {norm:.4f}")

V2 = Vt[:2, :]         # Shape (2, d)
A_proj = A_centered @ V2.T  # Shape (n, 2)

# Step 4: Plot the 2D projection
plt.figure(figsize=(10, 8))
plt.scatter(A_proj[:, 0], A_proj[:, 1], alpha=0.7)

# Annotate first 30 points with labels
for i in range(min(30, len(labels))):
    plt.text(A_proj[i, 0], A_proj[i, 1], str(labels[i]), fontsize=8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection onto Top 2 Principal Components (PCA)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()