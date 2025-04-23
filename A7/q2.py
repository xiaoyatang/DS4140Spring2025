import numpy as np
import pandas as pd
from numpy.linalg import eig
import matplotlib.pyplot as plt

# Load matrix
M = pd.read_csv("M.csv", header=None).values
n = M.shape[0]

# Compute "true" q_star using eigenvector method
eigvals, eigvecs = np.linalg.eig(M)
# index = np.argmin(np.abs(eigvals - 1))
q_true = np.real(eigvecs[:, 0])
q_true = np.abs(q_true)
q_true /= np.sum(q_true)

# Uniform initial state
q0 = np.ones(n) / n

# Track convergence
threshold = 1e-4
errors_matrix_power = []
errors_state_prop = []
ts = [2**i for i in range(11)]  # t = 2, 4, ..., 1024

# Matrix Power method
for t in ts:
    Mp = np.eye(n)
    for _ in range(t):
        Mp = Mp @ M
    q_mp = Mp @ q0
    q_mp /= np.sum(q_mp)
    errors_matrix_power.append(np.linalg.norm(q_mp - q_true, ord=1))

# State Propagation method
for t in ts:
    q = q0.copy()
    for _ in range(t):
        q = M @ q
    q /= np.sum(q)
    errors_state_prop.append(np.linalg.norm(q - q_true, ord=1))

# Find minimum t where error < threshold
def first_below(errors):
    for t, e in zip(ts, errors):
        if e < threshold:
            return t, e
    return None, None

t_mp, err_mp = first_below(errors_matrix_power)
t_sp, err_sp = first_below(errors_state_prop)

print(f"Matrix Power converged at t = {t_mp} with error = {err_mp:.2e}")
print(f"State Propagation converged at t = {t_sp} with error = {err_sp:.2e}")

# Optional plot
plt.plot(ts, errors_matrix_power, label="Matrix Power")
plt.plot(ts, errors_state_prop, label="State Propagation")
plt.axhline(threshold, color='gray', linestyle='--', label="Threshold")
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("t (iterations)")
plt.ylabel("L1 Error vs q_true")
plt.legend()
plt.title("Convergence vs t (Uniform Initial State)")
plt.grid(True)
plt.show()
