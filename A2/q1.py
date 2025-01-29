import requests
from collections import Counter
import sys

D1 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D1.txt'
D2 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D2.txt'
D3 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D3.txt'
D4 = 'https://users.cs.utah.edu/~jeffp/teaching/DM/A2/D4.txt'

docs = [requests.get(url).text for url in [D1, D2, D3, D4]]
#Construct 2-grams based on characters, for all documents.
def g1(text):
    bigrams = set()
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        bigrams.add(bigram)
    return len(bigrams),bigrams

# Construct 3-grams based on characters, for all documents.
def g2(text):
    trigrams = set()
    # trigrams = []
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        trigrams.add(trigram)
        # trigrams.append(trigram)
    return len(trigrams),trigrams

# bigram_counts = Counter(trigrams)

# # Find the number of distinct bigrams
# distinct_bigrams_count = len(bigram_counts)

# # Print the number of distinct bigrams
# print(f"There are {distinct_bigrams_count} distinct bigrams.")

# # Optionally, print each bigram and its count if the count is greater than 1
# for bigram, count in bigram_counts.items():
#     if count > 1:
#         print(f"Bigram '{bigram}' repeats {count} times.")


# Construct $2$-grams based on words, for all documents.  
def g3(text):
    words = text.split()
    bigrams = set()
    # bigrams = []
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigrams.add(bigram)
        # bigrams.append(bigram)
    return len(bigrams),bigrams

def Jaccard(set1, set2):
    """Calculate the Jaccard Similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0  # Avoid division by zero if both sets are empty
    return len(intersection) / len(union)

def main():
    # option = sys.argv[1]
    # if option == 'g1':
    #     func = g1
    # elif option == 'g2':
    #     func = g2
    # elif option == 'g3':
    #     func = g3
    # else:
    #     print("Invalid argument")
    #     return
    funcs = [g1, g2, g3]  
    func_names = ['g1', 'g2', 'g3']  

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

    # Print all k-grams collected, organized by document and k-gram type
    print("\nAll k-grams collected:")
    # for doc, types in all_k_grams.items():
    #     print(f"{doc}:")
    #     for type_name, info in types.items():
    #         print(f"  {type_name} - Length: {info['length']}")
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    print("\nJaccard Similarities:")
    for name in func_names:
        print(f"\n{name} similarities:")
        for (doc1, doc2) in pairs:
            set1 = all_k_grams[f'Document {doc1}'][name]['k_grams']
            set2 = all_k_grams[f'Document {doc2}'][name]['k_grams']
            similarity = Jaccard(set1, set2)
            print(f'Jaccard similarity between Document {doc1} and Document {doc2}: {similarity:.4f}')

if __name__ == '__main__':
    main()