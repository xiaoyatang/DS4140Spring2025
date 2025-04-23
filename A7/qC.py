import pandas as pd
import networkx as nx
import numpy as np

# Load M
M = pd.read_csv("M.csv", header=None).values
n = M.shape[0]

# Build directed graph from M
G = nx.DiGraph()
for i in range(n):
    for j in range(n):
        if M[j, i] > 0:  # Remember: M is column-stochastic
            G.add_edge(i, j)

# Check irreducibility
is_strongly_connected = nx.is_strongly_connected(G)

# Check aperiodicity (more complex, but here's a rough test)
aperiodic = nx.is_aperiodic(G)

print("Is the chain irreducible?", is_strongly_connected)
print("Is the chain aperiodic?", aperiodic)
print("Ergodic?" , is_strongly_connected and aperiodic)
