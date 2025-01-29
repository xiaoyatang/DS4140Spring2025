#!/bin/bash

# Define the list of function arguments
functions=('g1' 'g2' 'g3')

# Loop through each function and run the Python script with it
for func in "${functions[@]}"
do
    echo "Running $func..."
    python q1.py $func
done

echo "All functions have been executed."

