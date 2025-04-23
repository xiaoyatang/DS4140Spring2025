import hashlib
import requests

def download_data(url):
    response = requests.get(url)
    data = response.text
    return data

def simple_hash(seed):
    def hash_fn(x):
        return int(hashlib.md5(f"{seed}_{x}".encode()).hexdigest(), 16)
    return hash_fn

t = 5
k = 10
stream1 = download_data('https://users.cs.utah.edu/~jeffp/teaching/DM/A5/S1.txt')
stream2 = download_data('https://users.cs.utah.edu/~jeffp/teaching/DM/A5/S2.txt')
m1 = len(stream1)
m2 = len(stream2)

hash_functions = [simple_hash(i) for i in range(t)]
C = [[0] * k for _ in range(t)]

def count_min(stream, m):
    for char in stream:
        indices = [h(char) % k for h in hash_functions]
        for i in range(t):
            C[i][indices[i]] += 1
    results = {}
    for char in 'abc':
        estimated_count = min(C[i][h(char) % k] for i, h in enumerate(hash_functions))
        results[char] = (estimated_count, estimated_count / m)
    return results

results_s1 = count_min(stream1, m1)
results_s2 = count_min(stream2, m2)
print("Results for S1:", results_s1)
print("Results for S2:", results_s2)
