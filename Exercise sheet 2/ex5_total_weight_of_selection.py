# Exercise 5: How would you find the total weight of an individual's selection?

import numpy as np

if __name__ == "__main__":
    number_of_items = 1000

    # Example weights (normally you'd reuse the same weights for the whole problem)
    weights = np.random.uniform(low=1.0, high=10.0, size=number_of_items)

    # Example chromosome / selection
    selection_mask = np.random.choice([True, False], size=number_of_items)

    # Total weight of selected items
    total_weight = float(np.sum(weights[selection_mask]))

    print("Total selected weight:", total_weight)
