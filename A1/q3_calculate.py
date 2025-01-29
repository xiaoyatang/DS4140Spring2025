import math

def calculate_expression(n):
    result = n * (0.577 + math.log(n))
    return result

n = 300 
result = calculate_expression(n)
print("Result for n =", n, "is", result)

import numpy as np
m = 5000
k = np.sqrt(2*m)
print(k)