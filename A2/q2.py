import random
import hashlib

import requests
from collections import Counter
import sys

D1 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D1.txt'
D2 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D2.txt'


docs = [requests.get(url).text for url in [D1, D2]]

# Construct 3-grams based on characters, for all documents.
def g2(text):
    trigrams = set()
    # trigrams = []
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        trigrams.add(trigram)
        # trigrams.append(trigram)
    return len(trigrams),trigrams

def hash_function(seed):
    def hash(x):
        result = hashlib.sha256((str(seed) + x).encode()).hexdigest()
        return int(result, 16)
    return hash

def min_hash_signature(k_grams, num_hashes, m, seeds): # implementation of Algorithm 4.2.1 
    # Create a list of hash functions
    hash_funcs = [hash_function(seed) for seed in seeds]
    
    # Initialize the signature with infinity
    signature = [float('inf')] * num_hashes
    
    # Compute the min-hash signature
    for gram in k_grams:
        for i, func in enumerate(hash_funcs):
            hashed_value = func(gram) % m
            if hashed_value < signature[i]:
                signature[i] = hashed_value
    return signature

def estimate_jaccard(sig1, sig2):
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


funcs = [g2]  
func_names = ['g2']  

# Dictionary to store k-grams for all types and all documents
all_k_grams = {f'Document {i+1}': {} for i in range(len(docs))}

# Run each function on all documents
for func, name in zip(funcs, func_names):
    for idx, d in enumerate(docs):
        length, k_gram = func(d)
        all_k_grams[f'Document {idx + 1}'][name] = {
            'length': length,
            'k_grams': k_gram
        }
        print(f'{name} lengths for Document {idx + 1}:', length)

k_grams_d1 = all_k_grams['Document 1']['g2']['k_grams'] # g2 k_grams of D1
k_grams_d2 = all_k_grams['Document 2']['g2']['k_grams'] # g2 k_grams of D2

m = 10000  # Size of hash range

t_values = [20, 60, 150, 300, 600, 1200, 5000]
random_seed = 42
random.seed(random_seed)
seeds = [random.randint(1, 10000) for _ in range(max(t_values))]

results = {}
for t in t_values:
    sig1 = min_hash_signature(k_grams_d1, t, m, seeds[:t])
    sig2 = min_hash_signature(k_grams_d2, t, m, seeds[:t])
    jaccard_est = estimate_jaccard(sig1, sig2)
    results[t] = jaccard_est

for t, sim in results.items():
    print(f"t-value: {t}, Estimated Jaccard Similarity: {sim:.4f}")