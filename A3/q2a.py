'''q2a'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path_1 = "./R1-updated.csv"
file_path_2 = "./R2.csv"
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1) # calculate L2 norms
    return vectors / norms[:, np.newaxis]  # normalize all the data points

# compute angular similarity
def angular_similarity(a, b):
    dot_product = np.dot(a, b)
    # angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle numerical errors
    angle = np.arccos(dot_product)
    return 1 - (angle / np.pi)

# Normalize datasets
norm_df1 = normalize(df1.values)
norm_df2 = normalize(df2.values)

# Calculate angular similarities for all pairs within each dataset
def calculate_similarities(data):
    n = data.shape[0]
    similarities = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = angular_similarity(data[i], data[j])
            if not (0 <= sim <= 1):  # Check for computation errors
                print('Computation error detected!')
            similarities[(i, j)] = sim
    return similarities

similarities_dict_df1 = calculate_similarities(norm_df1)
similarities_dict_df2 = calculate_similarities(norm_df2)

tau = 0.60
def count_above_tau(similarities_dict, tau):
    return {pair: sim for pair, sim in similarities_dict.items() if sim > tau}

above_tau_df1 = count_above_tau(similarities_dict_df1, tau)
above_tau_df2 = count_above_tau(similarities_dict_df2, tau)

# Report results larger than tao=0.60
print(f"Pairs and their similarities (R1): {above_tau_df1}")
print(f"Pairs and their similarities (R2): {above_tau_df2}")
print(f"Number of pairs with angular similarity greater than {tau} in R1: {len(above_tau_df1)}")
print(f"Number of pairs with angular similarity greater than {tau} in R2: {len(above_tau_df2)}")

similarities_df1 = list(similarities_dict_df1.values())
similarities_df2 = list(similarities_dict_df2.values())

# Plot CDF
plt.hist(similarities_df1, bins=500, cumulative=True, density=True, histtype='step', label='R1')
plt.hist(similarities_df2, bins=500, cumulative=True, density=True, histtype='step', label='R2')

plt.title('CDF of Angular Similarities')
plt.xlabel('Angular Similarity')
plt.ylabel('Cumulative Density')
plt.grid(True)
plt.legend()
plt.show()



'''q2b'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
np.random.seed(42)

def generate_unit_vectors(t, d):
    # Generate random vectors with normal distribution
    random_vectors = np.random.randn(t, d)
    # Normalize 
    norms = np.linalg.norm(random_vectors, axis=1) # calculate L2 norms
    return random_vectors / norms[:, np.newaxis]  # normalize all the data points

def angular_similarity(a, b):
    dot_product = np.dot(a, b)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) 
    return 1 - (angle / np.pi)

def calculate_similarities(data):
    n = data.shape[0]
    similarities = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = angular_similarity(data[i], data[j])
            if not (0 <= sim <= 1):  # Check for computation errors
                print('Computation error detected!')
            similarities[(i, j)] = sim
    return similarities

def plot_cdf(data, tau):
    data_sorted = np.sort(data)
    yvals = np.arange(len(data_sorted)) / float(len(data_sorted))
    plt.plot(data_sorted, yvals, label='CDF')
    plt.axvline(x=tau, color='r', linestyle='--', label=f'Tau = {tau}')
    plt.title('CDF of Pairwise Angular Similarities')
    plt.xlabel('Angular Similarity')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

t = 250  # Number of vectors
d = 100  # Dimension of each vector
tau = 0.60  # Threshold for angular similarity

unit_vectors = generate_unit_vectors(t, d)
angular_similarities = calculate_similarities(unit_vectors)

def count_above_tau(similarities_dict, tau):
    return {pair: sim for pair, sim in similarities_dict.items() if sim > tau}

above_tau_df3 = count_above_tau(angular_similarities, tau)

# Report results larger than tao=0.60
print(f"Pairs and their similarities (randomly generated t values): {above_tau_df3}")
print(f"Number of pairs with angular similarity greater than {tau} in t random data points: {len(above_tau_df3)}")

similarities_df1 = list(angular_similarities.values())
plot_cdf(similarities_df1, tau)



'''q2c'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
np.random.seed(42)

ks_dist_R1_random = ks_2samp(similarities_df1, similarities_df1)
ks_dist_R2_random = ks_2samp(similarities_df2, similarities_df1)

print(f"KS distance between R1 and Random: {ks_dist_R1_random.statistic}")
print(f"KS distance between R2 and Random: {ks_dist_R2_random.statistic}")

# Determine which is more likely random
if ks_dist_R1_random.statistic < ks_dist_R2_random.statistic:
    print("R1 is more likely to be from a uniformly random distribution.")
else:
    print("R2 is more likely to be from a uniformly random distribution.")
