import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import requests
import io

# Load data from URLs
urls = [
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
]

datasets = [np.loadtxt(io.StringIO(requests.get(url).text), usecols=(1, 2)) for url in urls]
data = datasets[1]  # Using the second dataset as an example

def gonzalez_clustering(X, k, initial_index=0):
    n = len(X)
    centers = np.zeros((k, X.shape[1]))
    phi = np.zeros(n, dtype=int)
    M = np.full(n, np.inf)
    
    centers[0] = X[initial_index]
    distances = cdist(X, [centers[0]])
    M = distances.flatten()

    for i in range(1, k):
        next_center_index = np.argmax(M)
        centers[i] = X[next_center_index]
        new_distances = cdist(X, [centers[i]]).flatten()
        for j in range(n):
            if new_distances[j] < M[j]:
                M[j] = new_distances[j]
                phi[j] = i

    cost = np.sum(M**2)  # Compute the cost as the sum of squared distances
    return centers, phi, cost

# Determine the optimal number of clusters k
ks = range(1, 10)  # Testing k from 1 to 9
costs = []

for k in ks:
    _, _, cost = gonzalez_clustering(data, k)
    costs.append(cost)

# Plot the costs as a function of k to find the elbow
plt.figure(figsize=(10, 6))
plt.plot(ks, costs, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of clusters k')
plt.ylabel('Clustering Cost (Sum of Squared Distances)')
plt.xticks(ks)
plt.grid(True)
plt.show()
