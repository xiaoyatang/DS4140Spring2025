import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import requests
import io
# np.random.seed(42)

def k_means_plus_plus(X, k, initial_index=0):
    n = X.shape[0]
    centers = np.zeros((k, X.shape[1]))
    centers[0] = X[initial_index]

    distances = cdist(X, [centers[0]])
    D2 = distances.flatten()**2

    for i in range(1, k):
        probabilities = D2 / D2.sum()
        # cumulative_probabilities = np.cumsum(probabilities)
        # r = np.random.rand()
        # index = np.where(cumulative_probabilities >= r)[0][0]
        index = np.random.choice(n, p=probabilities)
        centers[i] = X[index]
        
        new_distances = cdist(X, [centers[i]]).flatten()**2
        D2 = np.minimum(D2, new_distances)

    final_distances = cdist(X, centers)
    assignments = np.argmin(final_distances, axis=1)

    return centers, assignments

def compute_k_means_cost(X, centers):
    distances = cdist(X, centers)
    min_distances = np.min(distances, axis=1)
    cost = np.sqrt(np.mean(min_distances**2))
    return cost

# def sort_centers(centers):
#     return centers[np.lexsort((centers[:, 1], centers[:, 0]))]

# def are_centers_same(centers1, centers2, tol=1e-5):
#     centers1_sorted = sort_centers(centers1)
#     centers2_sorted = sort_centers(centers2)
#     return np.allclose(centers1_sorted, centers2_sorted, atol=tol)

urls = [
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
]

datasets = [np.loadtxt(io.StringIO(requests.get(url).text), usecols=(1, 2)) for url in urls]
data = datasets[1]
k = 3
trials = 100
costs = []
results = []
same_assignments_count = 0
best_match_count = 0
best_centers = None
best_assignments = None

import pandas as pd

# Load the Gonzalez assignments
gonzalez_assignments = pd.read_csv('/Users/xiaoyatang/Desktop/tutorial_py/DS4140/DS4140Spring2025/A4/assignments.csv', header=None).values.flatten()

for _ in range(trials):
    centers, kpp_assignments  = k_means_plus_plus(data, k, initial_index=0)
    cost = compute_k_means_cost(data, centers)
    costs.append(cost)
    # results.append((centers, kpp_assignments))

    if np.array_equal(kpp_assignments, gonzalez_assignments):
        same_assignments_count += 1

    current_match_count = np.sum(kpp_assignments == gonzalez_assignments)
    if current_match_count > best_match_count:
        best_match_count = current_match_count
        best_centers = centers
        best_assignments = kpp_assignments

results.append((best_centers, best_assignments))
fraction_assignments_same = same_assignments_count / trials
print(f"Fraction of trials where all assignments are the same as Gonzalez: {fraction_assignments_same}")

costs = np.sort(costs)
plt.figure(figsize=(8, 6))
plt.plot(costs, np.linspace(0, 1, len(costs), endpoint=True), marker='o')
plt.title('Cumulative Density Function of 3-Means Costs')
plt.xlabel('3-Means Cost')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Randomly select one result for visualization
# random_index = np.random.randint(len(results))
selected_centers, selected_assignments = results[0]
# Plot the result
colors = ['blue', 'green', 'orange']  # Colors for the clusters
scolors = ['red', 'purple', 'gray']  # Colors for the centroids
plt.figure(figsize=(8, 6))

for i in range(k):
    cluster_points = data[selected_assignments == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
    plt.scatter(selected_centers[i, 0], selected_centers[i, 1], facecolors='none', edgecolors=scolors[i], s=200, marker='o', linewidths=2)

plt.title('Random k-Means++ Clustering Result')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()