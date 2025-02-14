import numpy as np

# Define the similarity matrix
similarity_matrix = np.array([
    [1.00, 0.72, 0.32, 0.35],
    [0.72, 1.00, 0.20, 0.55],
    [0.32, 0.20, 1.00, 0.89],
    [0.35, 0.55, 0.89, 1.00]
])

# Constants for the hash function (b and r) based on previous discussions
b = 8
r = 20

tau = 0.70

def f(s, b, r):
    return 1 - (1 - s**b)**r

# Compute the probability of being estimated as similar for each pair
probabilities = {}
labels = ['A', 'B', 'C', 'D']
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        pair_label = f"{labels[i]} & {labels[j]}"
        probabilities[pair_label] = f(similarity_matrix[i, j], b, r)

for pair, probability in probabilities.items():
    print(f"Probability that {pair} is estimated as similar: {probability:.4f}")
