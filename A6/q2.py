# import pandas as pd
# import numpy as np
# from scipy import linalg as LA

# # Load the distance matrix D
# D = pd.read_csv('D.csv')
# # Assume the first column contains labels (e.g., airports)
# labels = D.iloc[:, 0]        # This is a pandas Series of labels
# D = D.iloc[:, 1:].to_numpy(dtype=float)

# # Step 1: Square each element to get D^(2)
# D_squared = np.square(D)

# # Step 2: Compute Frobenius norm
# fro_norm = LA.norm(D_squared,'fro')

# print(f"Frobenius norm of D^(2): {fro_norm:.4f}")

# # Step 2: Create the centering matrix C_n
# n = D.shape[0]
# I = np.eye(n)
# ones = np.ones((n, n))
# C = I - (1/n) * ones  # This is C_n

# # Step 3: Double center the matrix
# M = -0.5 * C @ D_squared @ C

# # Step 4: Compute Frobenius norm of M
# fro_norm_M = LA.norm(M, 'fro')

# print(f"Frobenius norm of M: {fro_norm_M:.4f}")


# #eigendecomposition (forcing real eigenvalue squaroots)
# l,V = LA.eig(M)
# s = np.real(np.power(l,0.5))
# V2 = V[:,[0,1]]
# s2 = np.diag(s[0:2])
# #low (2) dimensiona points
# Q = V2 @ s2
# #printing
# import matplotlib.pyplot as plt 
# plt.plot(Q[:,0],Q[:,1],'ro') 
# plt.show()

import pandas as pd
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

# Load the distance matrix D
D_df = pd.read_csv('D.csv')
labels = D_df.iloc[:, 0]
D = D_df.iloc[:, 1:].to_numpy(dtype=float)

# Step 1: Compute D^(2)
D_squared = np.square(D)

# Step 2: Frobenius norm of D^(2)
fro_norm = LA.norm(D_squared, 'fro')
print(f"Frobenius norm of D^(2): {fro_norm:.4f}")

# Step 3: Create centering matrix C_n
n = D.shape[0]
I = np.eye(n)
ones = np.ones((n, n))
C = I - ones / n

# Step 4: Compute double-centered matrix M
M = -0.5 * C @ D_squared @ C

# Step 5: Frobenius norm of M
fro_norm_M = LA.norm(M, 'fro')
print(f"Frobenius norm of M: {fro_norm_M:.4f}")

# Step 6: Eigendecomposition of M
eigvals, eigvecs = LA.eigh(M)  # Use eigh since M is symmetric

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Keep top 2 components
top2_eigvals = eigvals[:2]
top2_eigvecs = eigvecs[:, :2]

# Step 7: Compute 2D points
Q = top2_eigvecs @ np.diag(np.sqrt(top2_eigvals))

# Step 8: Plotting
plt.figure(figsize=(10, 8))
plt.scatter(Q[:, 0], Q[:, 1], color='red', alpha=0.8)

# Annotate a few points
for i in range(min(30, len(labels))):
    plt.text(Q[i, 0], Q[i, 1], str(labels[i]), fontsize=8)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('MDS: 2D Projection from Distance Matrix')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
