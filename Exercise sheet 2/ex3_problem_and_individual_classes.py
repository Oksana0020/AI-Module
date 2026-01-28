# Exercise 3:
# Incorporate boolean chromosome creation into problem and individual class shells

import numpy as np

class SelectionProblem:
    """Problem container for selection-style GA problems."""
    def __init__(self, number_of_genes=1000, cost_function=None):
        self.number_of_genes = int(number_of_genes)
        self.cost_function = cost_function  # optional

class Individual:
    """Individual with boolean chromosome representing selection of items."""
    def __init__(self, problem: SelectionProblem):
        # Create a random individual: True = selected, False = not selected
        self.chromosome = np.random.choice([True, False], size=problem.number_of_genes)
        self.cost = None if problem.cost_function is None else float(problem.cost_function(self.chromosome))

    def crossover(self, other_parent):
        # Placeholder (implemented in Exercise 10 file)
        return None

    def mutate(self, mutation_rate, mutation_range=0):
        # Placeholder (implemented in Exercise 9 file)
        return None

if __name__ == "__main__":
    p = SelectionProblem()
    p.number_of_genes = 10

    ind1 = Individual(p)
    print(ind1.chromosome)
