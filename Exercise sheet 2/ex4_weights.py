# Exercise 4: Use np.random.uniform to create an array of 1000 floating point values between 1 and 10
# store in variable weights

import numpy as np

if __name__ == "__main__":
    number_of_items = 1000

    weights = np.random.uniform(low=1.0, high=10.0, size=number_of_items)

    print("weights length:", len(weights))
    print("min weight:", weights.min())
    print("max weight:", weights.max())
    print("first 10 weights:", weights[:10])
