# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# import requests
# import io

# urls = [
#     'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
#     'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
#     'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
#     'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
# ]

# datasets = [np.loadtxt(io.StringIO(requests.get(url).text), usecols=(1, 2)) for url in urls]  #return numpy arrays
# data = datasets[1]

# def gonzalez_clustering(X, k, initial_index=0):
#     n = len(X)
#     centers = np.zeros((k, X.shape[1]))
#     phi = np.zeros(n, dtype=int) # initialize the assignments
#     M = np.zeros(n)
    
#     # Initialize the first center to be the first data point
#     centers[0] = X[initial_index]
#     phi[:] = 0
#     M[:] = cdist(X, [centers[0]]).flatten()  # Initial distance to the first center

#     # Iteratively select the next centers
#     for i in range(1, k):
#         j = np.argmax(M)  # Select the point that is furthest away as the next center
#         # the final dp is an outlier so it is the second chosen center
#         centers[i] = X[j]
#         distances = cdist(X, [centers[i]]).flatten()  # update distances from the new centers
        
#         # Update the assignment if the new center is closer
#         update = distances < M
#         phi[update] = i
#         M[update] = distances[update]
        
#     return centers, phi

# # implement Gonzalez algorithm
# k = 3  # Number of clusters
# initial_index = 0  # Based on the requirement to start with the first point
# centers, assignments = gonzalez_clustering(data, k, initial_index)
# print(centers)

# colors = ['blue', 'green', 'orange']  # Colors for the clusters
# scolors = ['red', 'purple', 'gray']# Colors for the centroids
# plt.figure(figsize=(8, 6))

# for i in range(k):
#     cluster_points = data[assignments == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
#     plt.scatter(centers[i, 0], centers[i, 1], facecolors='none', edgecolors=scolors[i], s=200, marker='o', linewidths=2)  # Hollow centroids with colored edges
#     # plt.scatter(centers[i, 0], centers[i, 1], c=scolors[i], s=100, marker='x')

# plt.title('Gonzalez Clustering Results')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.grid(True)
# plt.show()
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
data = datasets[1]

def gonzalez_clustering(X, k, initial_index=0):
    n = len(X)
    centers = np.zeros((k, X.shape[1]))
    center_indices = np.zeros(k, dtype=int)
    phi = np.zeros(n, dtype=int)
    # M = np.full(n, np.inf)
    M = np.zeros(n)
    
    centers[0] = X[initial_index]
    center_indices[0] = initial_index
    distances = cdist(X, [centers[0]])
    M = distances.flatten()

    for i in range(1, k):
        next_center_index = np.argmax(M)
        centers[i] = X[next_center_index]
        center_indices[i] = next_center_index
        new_distances = cdist(X, [centers[i]]).flatten()
        for j in range(n):
            if new_distances[j] < M[j]:
                M[j] = new_distances[j]
                phi[j] = i

    return centers, center_indices, phi, M

k = 3
initial_index = 0
centers,center_indices, assignments, min_distances = gonzalez_clustering(data, k, initial_index)
print('centers:',center_indices,centers)
print('assignments:',assignments)
np.savetxt("assignments.csv", assignments, delimiter=",", fmt='%d')

# Compute the 3-center cost
three_center_cost = np.max(min_distances)
print(f"3-Center Cost: {three_center_cost}")

# Compute the 3-means cost
three_means_cost = np.sqrt(np.mean(min_distances**2))
print(f"3-Means Cost: {three_means_cost}")

# Plot results
colors = ['blue', 'green', 'orange']  # Colors for the clusters
scolors = ['red', 'purple', 'gray']  # Colors for the centroids
plt.figure(figsize=(8, 6))

for i in range(k):
    cluster_points = data[assignments == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centers[i, 0], centers[i, 1], facecolors='none', edgecolors=scolors[i], s=200, marker='o', linewidths=2)

plt.title('Gonzalez Clustering Results')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()
