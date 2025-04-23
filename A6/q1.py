import numpy as np
from scipy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

A = pd.read_csv('A.csv')
A = A.iloc[:, 1:].to_numpy(dtype=float)
# a=np.random.rand(3,2)
# print(type(a))
U, s, Vt = LA.svd(A, full_matrices=False)
# print(U.shape,'U',U)
# print(s.shape,'s',s)
# print(Vt.shape,'Vt',Vt)
s = np.diag(s)
Ak = list()
l2_norms = []
for k in range(1,51):
    Ak_k = U[:,:k] @ s[:k,:k]@Vt[:k,:]
    Ak.append(Ak_k)
    norm_diff = LA.norm(A - Ak_k, 2)
    l2_norms.append(norm_diff)

print("k | L2 Norm")
for k, norm in enumerate(l2_norms, start=1):
    print(f"{k} | {norm:.4f}")

norm_A = LA.norm(A, 2)
print("L2 Norm of A:", norm_A)

# Take the top 2 right singular vectors (i.e., first 2 rows of Vt)
Vt2 = Vt[:2, :]  # shape (2, n)

# Project A onto top 2 right singular vectors
A_proj_2d = A @ Vt2.T  # shape (m, 2)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(A_proj_2d[:, 0], A_proj_2d[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection of A onto Top 2 Right Singular Vectors')
plt.grid(True)
plt.axis('equal')
plt.show() 