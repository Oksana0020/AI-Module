import numpy as np
from copy import deepcopy

# -------------------------------------------------
# Name: Sphere Function (minimum at all zeros)
# Goal:
# Minimise the classic sphere function:
#     f(x) = sum(x_i^2)
# The minimum value is 0, which occurs when every gene is 0:
#     x = [0, 0, 0, ..., 0]
# -------------------------------------------------

def sphere_cost(candidate_vector: np.ndarray) -> float:
    """Sphere cost function: sum of squared gene values."""
    return float(np.sum(candidate_vector ** 2))


class SphereProblem:
    """Defines the sphere optimisation problem for the GA."""
    def __init__(self, number_of_genes: int = 8):
        self.number_of_genes = int(number_of_genes)
        self.min_gene_value = -10
        self.max_gene_value = 10
        self.cost_function = sphere_cost


class Individual:
    """One candidate solution (chromosome) plus its cost"""
    def __init__(self, problem: SphereProblem):
        # Random real-valued chromosome within allowed bounds
        self.chromosome = np.random.uniform(
            low=problem.min_gene_value,
            high=problem.max_gene_value,
            size=problem.number_of_genes
        )
        # Evaluate how good/bad this solution is
        self.cost = float(problem.cost_function(self.chromosome))


class GAParameters:
    """Hyperparameters controlling GA behaviour."""
    def __init__(self):
        self.population_size = 300
        self.generations = 300
        self.child_rate = 0.5  # fraction of population size created as children each generation


def tournament_selection(population: list[Individual], tournament_size: int = 3) -> Individual:
    """
    Tournament selection:
    - Randomly select `tournament_size` individuals.
    - Return the one with the lowest cost.
    """
    candidates = np.random.choice(population, size=tournament_size, replace=False)
    best_candidate = candidates[0]

    for candidate in candidates[1:]:
        if candidate.cost < best_candidate.cost:
            best_candidate = candidate

    return best_candidate


def uniform_crossover(parent_a: Individual, parent_b: Individual) -> np.ndarray:
    """
    Uniform crossover:
    Each gene is taken from either parent with 50% probability
    """
    gene_mask = np.random.rand(len(parent_a.chromosome)) < 0.5
    child_vector = parent_a.chromosome.copy()
    child_vector[gene_mask] = parent_b.chromosome[gene_mask]
    return child_vector


def gaussian_mutation(child_vector: np.ndarray,
                      problem: SphereProblem,
                      mutation_rate: float = 0.1,
                      sigma: float = 0.3) -> np.ndarray:
    """
    Gaussian mutation:
    - Each gene mutates with probability `mutation_rate`.
    - Mutation adds N(0, sigma) noise.
    - Clamp results to allowed bounds.
    """
    mutated = child_vector.copy()

    for gene_index in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[gene_index] += np.random.normal(loc=0.0, scale=sigma)
            mutated[gene_index] = np.clip(
                mutated[gene_index],
                problem.min_gene_value,
                problem.max_gene_value
            )

    return mutated


def run_genetic_algorithm(problem: SphereProblem,
                          params: GAParameters,
                          mutation_rate: float = 0.1,
                          sigma: float = 0.3,
                          tournament_size: int = 3):
    """
    Main GA loop:
    1) Initialise population.
    2) Repeat for N generations:
       - Select parents
       - Crossover
       - Mutate
       - Merge + sort + cull
       - Track best solution found
    """
    population_size = params.population_size
    number_of_children = int(params.child_rate * population_size)

    # Initialise population
    population = [Individual(problem) for _ in range(population_size)]
    population.sort(key=lambda ind: ind.cost)
    best_solution = deepcopy(population[0])

    # Evolution loop
    for _ in range(params.generations):
        children: list[Individual] = []

        # Create children until the required number is produced
        while len(children) < number_of_children:
            parent_1 = tournament_selection(population, tournament_size=tournament_size)
            parent_2 = tournament_selection(population, tournament_size=tournament_size)

            child_vector = uniform_crossover(parent_1, parent_2)
            child_vector = gaussian_mutation(child_vector, problem,
                                             mutation_rate=mutation_rate,
                                             sigma=sigma)

            child = Individual(problem)
            child.chromosome = child_vector
            child.cost = float(problem.cost_function(child.chromosome))
            children.append(child)

        # Merge children into population and keep the best individuals
        population.extend(children)
        population.sort(key=lambda ind: ind.cost)
        population = population[:population_size]

        # Track best-ever solution
        if population[0].cost < best_solution.cost:
            best_solution = deepcopy(population[0])

    return population, best_solution


if __name__ == "__main__":
    # Number of genes can be changed depending on the problem size
    number_of_genes = 8
    problem = SphereProblem(number_of_genes=number_of_genes)
    params = GAParameters()
    final_population, best = run_genetic_algorithm(problem, params)
    print("Best chromosome:", np.round(best.chromosome, 6))
    print("Best cost:", best.cost)
