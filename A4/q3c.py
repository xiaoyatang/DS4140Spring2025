import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import requests
import io
np.random.seed(42)

def k_means_plus_plus(X, k, initial_index=0):
    n = X.shape[0]
    centers = np.zeros((k, X.shape[1]))
    center_indices = np.zeros(k, dtype=int)
    centers[0] = X[initial_index]
    center_indices[0] = initial_index

    distances = cdist(X, [centers[0]])
    D2 = distances.flatten()**2

    for i in range(1, k):
        probabilities = D2 / D2.sum()
        # cumulative_probabilities = np.cumsum(probabilities)
        # r = np.random.rand()
        # index = np.where(cumulative_probabilities >= r)[0][0]
        index = np.random.choice(n, p=probabilities)
        centers[i] = X[index]
        center_indices[i] = index

        new_distances = cdist(X, [centers[i]]).flatten()**2
        D2 = np.minimum(D2, new_distances)

    final_distances = cdist(X, centers)
    assignments = np.argmin(final_distances, axis=1)

    return centers, center_indices, assignments

def compute_k_means_cost(X, centers):
    distances = cdist(X, centers)
    min_distances = np.min(distances, axis=1)
    cost = np.sqrt(np.mean(min_distances**2))
    return cost


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


urls = [
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
]

datasets = [np.loadtxt(io.StringIO(requests.get(url).text), usecols=(1, 2)) for url in urls]
data = datasets[1]
k = 3
trials = 200
costs = []
same_assignments_count = 0

for _ in range(trials):
    initial_centers, initial_indices,kpp_assignments  = k_means_plus_plus(data, k, initial_index=0)
    # Run Lloyd's Algorithm
    final_centers, final_assignments = lloyds_algorithm(data, initial_indices)
    final_cost = compute_k_means_cost(data, final_centers)
    costs.append(final_cost)
    if np.array_equal(kpp_assignments, final_assignments):
        same_assignments_count += 1

# Report the fraction of same assignments
fraction_same = same_assignments_count / trials
print(f"Fraction of trials with the same assignments: {fraction_same:.2f}")
print(costs)
costs = np.sort(costs)
plt.figure(figsize=(8, 6))
plt.plot(costs, np.linspace(0, 1, len(costs), endpoint=True), marker='o')
plt.title('Cumulative Density Function of 3-Means Costs')
plt.xlabel('3-Means Cost')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()




