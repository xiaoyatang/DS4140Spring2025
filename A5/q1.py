import requests

def download_data(url):
    response = requests.get(url)
    data = response.text
    return data

def misra_gries(stream, k):
    counters = {}
    for element in stream:
        if element in counters: # if element has been counted, increment the count
            counters[element] += 1
        elif len(counters) < k: # if no match and a counter is 0, reassign the label 
            counters[element] = 1
        else:
            for key in list(counters.keys()): # decrement all counters.
                counters[key] -= 1
                if counters[key] == 0:
                    del counters[key]
    return counters

stream1 = download_data('https://users.cs.utah.edu/~jeffp/teaching/DM/A5/S1.txt')
stream2 = download_data('https://users.cs.utah.edu/~jeffp/teaching/DM/A5/S2.txt')

k = 9
counters_s1 = misra_gries(stream1, k)
counters_s2 = misra_gries(stream2, k)

# Calculating estimated ratios
m1 = len(stream1)
m2 = len(stream2)
print('c1:',counters_s1)
print('c1:',counters_s2)

ratios_s1 = {char: count / m1 for char, count in counters_s1.items()}
ratios_s2 = {char: count / m2 for char, count in counters_s2.items()}
print('s1:',ratios_s1)
print('s2:',ratios_s2)