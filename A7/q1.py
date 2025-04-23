import numpy as np
import pandas as pd

# Load matrix M from a CSV file
M = pd.read_csv("M.csv", header=None).values

# Initialize q_0
n = M.shape[0]
q0 = np.zeros(n)
q0[0] = 1

t = 1024

def matrix_power_naive(M, t):
    result = M.copy()
    for _ in range(t - 1):
        result = result @ M
    return result

def matrix_power_exponentiation(M, t):
    result = np.eye(M.shape[0])
    while t > 0:
        if t % 2 == 1:
            result = result @ M
        M = M @ M
        t //= 2
    return result

## matrix power
# # Naive method
# M_power_naive = matrix_power_naive(M, t)
# q_star_naive = M_power_naive @ q0

# # Efficient method
# M_power_exp = matrix_power_exponentiation(M, t)
# q_star_exp = M_power_exp @ q0

# # Display results
# print("q* using naive method:")
# print(q_star_naive)

# print("\nq* using exponentiation by squaring:")
# print(q_star_exp)

# # You can optionally also compute difference
# print("\nDifference between the two methods:")
# print(np.linalg.norm(q_star_naive - q_star_exp))


## State propagation
# for i in range(t):
#     if i==0:
#         q=q0
#     q = M @ q

# # Report result
# print("q* using state propagation after 1024 iterations:")
# print(q)

# # Optionally plot convergence (if needed)
# import matplotlib.pyplot as plt
# plt.plot(q, marker='o')
# plt.title("q* after 1024 State Propagation Steps")
# plt.xlabel("State Index")
# plt.ylabel("Probability")
# plt.grid(True)
# plt.show()

## Random walk
# # Choose a starting state (say state i=0)
# start_state = 0

# # Number of steps
# t0 = 100    # burn-in steps
# t = 1024    # sampling steps

# # Ensure columns are probabilities (column-stochastic matrix)
# assert np.allclose(M.sum(axis=0), 1), "Matrix M must be column-stochastic."

# # Precompute cumulative probabilities for sampling
# cumulative_M = np.cumsum(M, axis=0)

# def sample_from_column(col):
#     r = np.random.rand()
#     return np.searchsorted(cumulative_M[:, col], r)

# # Burn-in phase
# state = start_state
# for _ in range(t0):
#     state = sample_from_column(state)

# # Sampling phase
# visit_counts = np.zeros(n)
# for _ in range(t):
#     visit_counts[state] += 1
#     state = sample_from_column(state)

# # Normalize to get q_*
# q_star = visit_counts / np.sum(visit_counts)

# # Report result
# print("Estimated q* from random walk:")
# print(q_star)

# # Optionally plot
# import matplotlib.pyplot as plt
# plt.bar(np.arange(n), q_star)
# plt.title("Estimated q* from Random Walk")
# plt.xlabel("State Index")
# plt.ylabel("Estimated Probability")
# plt.grid(True)
# plt.show()

## Eigenvalue
from numpy.linalg import eig
eigvals, eigvecs = eig(M)


# Extract the corresponding eigenvector
v = np.real(eigvecs[:, 0])  # Ensure it's real

# Normalize to make it a probability distribution (L1 normalization)
q_star = v / np.sum(v)

# Display result
print("q* from dominant eigenvector (L1-normalized):")
print(q_star)

# Optional: plot
import matplotlib.pyplot as plt
plt.bar(np.arange(len(q_star)), q_star)
plt.title("q* from Eigenvector")
plt.xlabel("State Index")
plt.ylabel("Probability")
plt.grid(True)
plt.show()