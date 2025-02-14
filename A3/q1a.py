import numpy as np
import matplotlib.pyplot as plt

t = 160
# Threshold for Jaccard similarity
tau = 0.70
s_values = np.linspace(0, 1, 100)

def f(s, b, r):
    return 1 - (1 - s**b)**r

# Possible values for b and r where b * r = t
possible_values = [(b, t // b) for b in range(1, t + 1) if t % b == 0]

plt.figure(figsize=(10, 6))
for b, r in possible_values:
    f_values = f(s_values, b, r)
    plt.plot(s_values, f_values, label=f"b={b}, r={r}")

plt.axvline(x=tau, color='red', linestyle='--', label=f"Tau = {tau}")
plt.title("S-curve for Different Combinations of b and r")
plt.xlabel("Jaccard Similarity s")
plt.ylabel("Probability f(s)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

