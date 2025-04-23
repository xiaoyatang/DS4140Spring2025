import numpy as np
import pandas as pd

# Load M and convert to row-stochastic if needed
M = pd.read_csv("M.csv", header=None).values
M = M.T  # Assuming your matrix is column-stochastic, convert to row-stochastic

# Initial state: start at node 3
q = np.zeros(M.shape[0])
q[3] = 1

# Dozen steps (12 iterations)
for _ in range(12):
    q = M @ q

# Result: probability of being in node 0
prob_node_0 = q[0]

print(f"Probability of being in node 0 after 12 steps from node 3: {prob_node_0:.6f}")
