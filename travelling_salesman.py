# Constants, experiment parameters
import itertools
from scipy import special as sc
import time
from weighted_graph import *
import csv
import random
NUM_CITIES = 4
POPULATION_SIZE = 10
MIXING_NUMBER = 2
MUTATION_RATE = 0.01


# Create the fitness score - How good is a solution?
def fitness_score(seq, distance_matrix):
    total_distance = 0
    try:
        for i in range(NUM_CITIES - 1):
            city1 = seq[i]
            city2 = seq[i+1]
            if city2 in distance_matrix[city1]:
                distance_between_cities = distance_matrix[city1][city2]
                total_distance += distance_between_cities
            else:
                continue
        # total_distance += distance_matrix[seq[-1]][seq[0]]
        return total_distance
    except Exception as error:
        print(error)


# Create the selection operator acording their fitness score
# Select best solutions for next step: crossover


def selection(population, distance_matrix):
    parents = []

    for ind in population:
        # select parents with probability proportional to their fitness score
        if random.randint(0, sc.comb(NUM_CITIES, 2) * 2 - 1) < fitness_score(ind, distance_matrix):
            parents.append(ind)

    return parents


# Create the crossover operator
# Combine features of each solution using a crossover point


def crossover(parents):

    # random indexes to to cross states with
    cross_points = random.sample(range(NUM_CITIES), MIXING_NUMBER - 1)
    offsprings = []

    # all permutations of parents
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))

    for perm in permutations:
        offspring = []

        # track starting index of sublist
        start_pt = 0

        # doesn't account for last parent
        for parent_idx, cross_point in enumerate(cross_points):

            # sublist of parent to be crossed
            parent_part = perm[parent_idx][start_pt:cross_point]
            offspring.append(parent_part)

            # update index pointer
            start_pt = cross_point

        # last parent
        last_parent = perm[-1]
        parent_part = last_parent[cross_point:]
        offspring.append(parent_part)

        # flatten the list since append works kinda differently
        offsprings.append(list(itertools.chain(*offspring)))

    return offsprings


# Create the routine to mutate a solution
# A operator to create diversity in the population
def mutate(seq, cities):
    for row in range(len(seq)):
        if random.random() < MUTATION_RATE:
            # Escolhe uma cidade aleatória do conjunto
            new_city = random.choice(cities)
            seq[row] = new_city

    return seq

# Print the solution


def print_found_goal(population, distance_matrix, to_print=True):
    for ind in population:
        score = fitness_score(ind, distance_matrix)
        if to_print:
            print(f'{ind}. Score: {score}')
        if score == sc.comb(NUM_CITIES, 2):
            if to_print:
                print('Solution found')
            return True

    if to_print:
        print('Solution not found')
    return False


# Create the routine to implement the evolution
def evolution(population, distance_matrix, fitness_score, cities):
    # select individuals to become parents
    parents = selection(population, distance_matrix)

    # recombination. Create new offsprings
    offsprings = crossover(parents)

    # mutation
    offsprings = list(map(mutate, offsprings, cities))

    # introduce top-scoring individuals from the previous generation and keep top fitness individuals
    new_gen = offsprings + population

    # Utilize a matriz de distâncias na avaliação de aptidão
    new_gen = sorted(new_gen, key=lambda ind: fitness_score(
        ind, distance_matrix), reverse=True)[:POPULATION_SIZE]

    return new_gen
# Running the experiment


generation = 0

# Create the initial population (solutions)


def generate_population(graph):
    population = []
    cities = ['Ouro Preto', 'Mariana', 'Alvinopolis', 'BH']

    # Criar uma matriz de distâncias entre as cidades
    distance_matrix = {
        'Ouro Preto': {'Mariana': 14.4, 'BH': 101.5},
        'Mariana': {'Ouro Preto': 14.4, 'Alvinopolis': 69.3},
        'Alvinopolis': {'Mariana': 69.3, 'BH': 162},
        'BH': {'Ouro Preto': 69.3, 'Alvinopolis': 162}
    }

    for i in range(POPULATION_SIZE):
        # Criar uma rota aleatória inicial
        route = random.sample(cities, len(cities))
        population.append(route)

    return population, distance_matrix


def save_to_csv(generation, score, time, num_pop, mutation_rate, n_queens):
    with open('results_salesman.csv', mode='a', newline=f'\n') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['generation',
                            'score', 'time', 'num_pop', 'mutation_rate', 'n_queens'])
        writer.writerow([generation,
                        score, time, num_pop, mutation_rate, n_queens])


cities_graph = Weighted_Graph()
cities = ['Ouro Preto', 'Mariana', 'Alvinopolis', 'BH']

# Generate Random Population
population, distance_matrix = generate_population(cities_graph)

initial_time = time.time()

# Generations until found the solution
while not print_found_goal(population, distance_matrix):
    print(f'Generation: {generation}')
    print_found_goal(population, distance_matrix)
    population = evolution(population, distance_matrix, fitness_score, cities)
    generation += 1

end_time = time.time()
save_to_csv(generation, '?', round((end_time - initial_time), 2),
            POPULATION_SIZE, MUTATION_RATE, NUM_CITIES)

print('Tempo gasto: ', (end_time - initial_time))
