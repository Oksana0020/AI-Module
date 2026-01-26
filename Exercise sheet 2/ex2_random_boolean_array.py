# sheet2_ex2_random_boolean_array.py
# Exercise Sheet 2 — Exercise 2
#
# Exercise 2: Use np.random.choice to create an array of size 1000 of True or False values

import numpy as np

if __name__ == "__main__":
    number_of_genes = 1000

    # Random chromosome (True/False)
    chromosome = np.random.choice([True, False], size=number_of_genes)

    print("Chromosome length:", len(chromosome))
    print("Chromosome dtype:", chromosome.dtype)
    print("First 25 genes:", chromosome[:25])
