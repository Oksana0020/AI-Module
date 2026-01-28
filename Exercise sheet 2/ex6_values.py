# Exercise 6: Use np.random.randint to create an array of 1000 integer values between 0 and 50
# store in variable values

import numpy as np

if __name__ == "__main__":
    number_of_items = 1000

    # randint high is exclusive, so use 51 to include 50
    values = np.random.randint(low=0, high=51, size=number_of_items)

    print("values length:", len(values))
    print("min value:", values.min())
    print("max value:", values.max())
    print("first 10 values:", values[:10])
