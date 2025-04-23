import numpy as np
import requests
import io
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

urls = [
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C1.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C2.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C3.txt',
    'https://users.cs.utah.edu/~jeffp/teaching/DM/A4/C4.txt'
]

datasets = [np.loadtxt(io.StringIO(requests.get(url).text)) for url in urls]  #return numpy arrays


def perform_clustering(data, indices):
    # Separate indices for use in annotations
    point_indices = indices.astype(int)
    # Separate coordinates for clustering
    coordinates = data[:, 1:3]

    # Perform single-link hierarchical clustering
    single_linkage = linkage(coordinates, method='single')
    clusters_single = fcluster(single_linkage, t=4, criterion='maxclust')

    # Perform complete-link hierarchical clustering
    complete_linkage = linkage(coordinates, method='complete')
    clusters_complete = fcluster(complete_linkage, t=4, criterion='maxclust')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=clusters_single, cmap='magma', marker='o', edgecolor='k')
    plt.title('Single-Link Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.colorbar(label='Cluster ID')

    for i, txt in enumerate(point_indices):
        plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]))

    plt.subplot(1, 2, 2)  
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=clusters_complete, cmap='magma', marker='o', edgecolor='k')
    plt.title('Complete-Link Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.colorbar(label='Cluster ID')

    for i, txt in enumerate(point_indices):
        plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]))

    plt.tight_layout()
    plt.show()
    return clusters_single,clusters_complete

sin,com= perform_clustering(datasets[0], datasets[0][:, 0])
print("Single:",sin,"\n","Complete:",com)