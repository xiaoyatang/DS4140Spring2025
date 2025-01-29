import random

def generate(space=300):
    seen_numbers = set()
    k = 0
    while len(seen_numbers) < space:
        i = random.randint(0, space-1)
        k += 1
        seen_numbers.add(i)
    return k

# for q2-a
# Example usage:
# trial_result = generate(300)
# print('Number of trials until each number has been generated at least once:', trial_result)


def main(count, space):
    results = []
    for i in range(count):
        random.seed(i)  # Set a different seed for each trial
        trials = generate(space)
        results.append(trials)
    return results


import matplotlib.pyplot as plt
def cdf(data):
    """Calculates the empirical CDF of a given dataset."""
    # sorted the data
    sorted_data = sorted(data)
    # Calculate the length of the data
    n = len(sorted_data)
    # Create a list of CDF values
    cdf_values = []
    for i, x in enumerate(sorted_data):
        cdf_values.append((i+1) / n)
    return sorted_data, cdf_values
# for q2-b
# if __name__ == '__main__':
#     # Run the simulation 100 times
#     trial_results = main(400, 300)
#     average_trials = sum(trial_results) / len(trial_results)
#     print('Empirical expectation of k:', average_trials)
#     x, y = cdf(trial_results)
#     plt.step(x, y, where="post")
#     plt.xlabel("K trials")
#     plt.ylabel("CDF")
#     plt.title("Empirical CDF")
#     plt.show()

# for q2-d
import time

def simulate_trials(m, n):
    start_time = time.time()
    trials_counts = [generate(n) for _ in range(m)]
    duration = time.time() - start_time
    average_trials = sum(trials_counts) / m
    return duration, average_trials

m_values = [400, 1000, 2000, 4000, 5000]  # Fixed values for m
n_values = [300, 5000, 10000, 20000]  # Increasing values for n

# Collect results for plotting
results = {m: {'durations': [], 'averages': []} for m in m_values}

# Perform simulations
for m in m_values:
    for n in n_values:
        duration, avg_trials = simulate_trials(m, n)
        results[m]['durations'].append(duration)
        results[m]['averages'].append(avg_trials)
        print(f"Completed m={m}, n={n} in {duration:.2f}s with average trials {avg_trials:.2f}")

# Plotting 
plt.figure(figsize=(10, 6))
for m in m_values:
    plt.plot(n_values, results[m]['durations'], label=f'm = {m}', marker='o')

plt.xlabel("n (size of space)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime as a Function of n for Different Values of m")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()

