import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import requests
import io

urls = [
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
]

datasets = [np.loadtxt(io.StringIO(requests.get(url).text), usecols=(1, 2)) for url in urls]
data = datasets[1]

def lloyds_algorithm(X, initial_centers):
    k = len(initial_centers)
    centers = X[initial_centers]  # Initialize centers with points at indices 1, 2, 3 (zero-indexed)
    assignments = np.zeros(len(X), dtype=int)
    
    while True:
        old_assignments = assignments.copy()
        
        # Step 1: Assign each x to the closest center
        distances = cdist(X, centers)
        assignments = np.argmin(distances, axis=1)
        
        # Step 2: Update each center to be the mean of points assigned to it
        for i in range(k):
            if np.any(assignments == i):
                centers[i] = np.mean(X[assignments == i], axis=0)
        
        # Check for convergence (if assignments do not change, break)
        if np.array_equal(old_assignments, assignments):
            break
    
    return centers, assignments

def compute_k_means_cost(X, centers, assignments):
    distances = cdist(X, centers)
    min_distances = np.min(distances, axis=1)
    cost = np.sqrt(np.mean(min_distances**2))
    return cost

initial_indices = [0, 1, 2]  # Python uses zero-indexing, corresponding to indices 1, 2, 3 in a one-indexed system

# Run Lloyd's Algorithm
final_centers, final_assignments = lloyds_algorithm(data, initial_indices)

# Compute the k-means cost
final_cost = compute_k_means_cost(data, final_centers, final_assignments)
print("Final Centers:\n", final_centers)
print("3-Means Cost:", final_cost)

# Plotting the results
plt.scatter(data[:, 0], data[:, 1], c=final_assignments, cmap='viridis', alpha=0.5)
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', s=200, marker='x')
plt.title('Lloyd\'s Algorithm Results')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
