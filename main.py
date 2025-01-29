import random
import math
import numpy as np
from colorama import Fore, init, Style

init()


# define sphere function
def sphere_function(x):
    return sum(xi**2 for xi in x)


# define function for getting neighbors of a current point
def get_neighbors(x, epsilon):
    x1, x2 = x
    return [
        (x1 + epsilon, x2),
        (x1 - epsilon, x2),
        (x1, x2 + epsilon),
        (x1, x2 - epsilon),
    ]


# define hill climbing algorithm
def hill_climbing(func, point, iterations=1000, epsilon=1e-2):
    current_point = point
    current_value = func(current_point)

    for i in range(iterations):
        neighbors = get_neighbors(current_point, epsilon)

        # find the best neighbor
        next_point = None
        next_value = np.inf

        for neighboor in neighbors:
            value = func(neighboor)
            if value < next_value:
                next_point = neighboor
                next_value = value

        # if there is no better neighbor, break the loop
        if next_value >= current_value:
            break

        # move to the best neighbor
        current_point, current_value = next_point, next_value

    return current_point, current_value


# define function for getting random neighbor
def get_random_neighbor(x, epsilon):
    x1, x2 = x
    return (
        x1 + random.uniform(-epsilon, epsilon),
        x2 + random.uniform(-epsilon, epsilon),
    )


# define random local search algorithm
def random_local_search(func, point, iterations=1000, epsilon=1e-2, probability=0.2):
    current_point = point
    current_value = func(current_point)

    for i in range(iterations):
        # get random neighbor
        new_point = get_random_neighbor(current_point, epsilon)
        new_value = func(new_point)

        # check if change a point
        if new_value < current_value or random.random() < probability:
            current_point, current_value = new_point, new_value

    return current_point, current_value


# define simulated annealing algorithm
def simulated_annealing(
    func, point, iterations=1000, epsilon=1e-2, temperature=1000, cooling_rate=0.85
):
    current_solution = point
    current_energy = func(current_solution)

    best_solution = None
    best_energy = float("inf")

    for i in range(iterations):
        while temperature > 0.001:
            # get random neighbor
            new_soluttion = get_random_neighbor(current_solution, epsilon)
            new_energy = func(new_soluttion)
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.random() < math.exp(
                -delta_energy / temperature
            ):
                current_solution, current_energy = new_soluttion, new_energy

            temperature *= cooling_rate

        if current_energy < best_energy:
            best_solution, best_energy = current_solution, current_energy

    return best_solution, best_energy


if __name__ == "__main__":
    # function bounds
    bounds = [(-5, 5), (-5, 5)]

    # define epsilon range
    epsilon_range = [
        1e-1,
        1e-2,
        1e-3,
    ]

    # define random current point
    random_initial_point = (
        random.uniform(bounds[0][0], bounds[0][1]),
        random.uniform(bounds[1][0], bounds[1][1]),
    )

    for epsilon in epsilon_range:
        print("=======================================================================")
        print(Fore.GREEN + "Results with epsilon:", epsilon, Style.RESET_ALL)
        print("Hill Climbing Algorithm:")
        hc_solution, hc_value = hill_climbing(
            sphere_function, random_initial_point, epsilon=epsilon
        )
        print("Solution:", hc_solution, "Value:", Fore.BLUE, hc_value, Style.RESET_ALL)

        print("\nRandom Local Search Algorithm:")
        rls_solution, rls_value = random_local_search(
            sphere_function, random_initial_point, epsilon=epsilon
        )
        print(
            "Solution:", rls_solution, "Value:", Fore.BLUE, rls_value, Style.RESET_ALL
        )

        print("\nSimulated Annealing Algorithm:")
        sa_solution, sa_value = simulated_annealing(
            sphere_function, random_initial_point, epsilon=epsilon
        )
        print("Solution:", sa_solution, "Value:", Fore.BLUE, sa_value, Style.RESET_ALL)
        print("=======================================================================")

    # define initial point
    initial_point = (2, 2)

    # algorithms execution with predifined point close to the global minima
    print("=======================================================================")
    print(
        Fore.GREEN
        + "Results with predifined initial point (2, 2) close to the global minima:"
        + Style.RESET_ALL
    )
    print("Hill Climbing Algorithm:")
    hc_solution, hc_value = hill_climbing(sphere_function, initial_point)
    print("Solution:", hc_solution, "Value:", Fore.BLUE, hc_value, Style.RESET_ALL)

    print("\nRandom Local Search Algorithm:")
    rls_solution, rls_value = random_local_search(sphere_function, initial_point)
    print("Solution:", rls_solution, "Value:", Fore.BLUE, rls_value, Style.RESET_ALL)

    print("\nSimulated Annealing Algorithm:")
    sa_solution, sa_value = simulated_annealing(sphere_function, initial_point)
    print("Solution:", sa_solution, "Value:", Fore.BLUE, sa_value, Style.RESET_ALL)
    print("=======================================================================")
