# Constants, experiment parameters
import itertools
from scipy import special as sc
import time
from weighted_graph import *
from itertools import chain
import csv
import random
NUM_CITIES = 10
POPULATION_SIZE = 10
MIXING_NUMBER = 2
MUTATION_RATE = 0.1
MAX_GEN = 1000

# ta puro


def fitness_score(seq, distance_matrix):
    total_distance = 0

    try:

        # Calcula a distância acumulada entre nós na sequência
        for i in range(len(seq) - 1):
            city1 = seq[i]
            city2 = seq[i + 1]

            # Verifica se há um caminho entre as cidades na matriz de distâncias
            if city2 in distance_matrix.get(city1, {}):
                distance_between_cities = distance_matrix[city1][city2]
                total_distance += distance_between_cities
            else:
                # Se não houver caminho, a sequência é inválida
                return 347.2

        return total_distance

    except Exception as error:
        print(error)

# Create the selection operator acording their fitness score
# Select best solutions for next step: crossover

# nao sei se precisa consertar


def selection(population, distance_matrix):
    parents = []

    for ind in population:
        # select parents with probability proportional to their fitness score
        if random.randint(0, sc.comb(NUM_CITIES, 2) * 2 - 1) < fitness_score(ind, distance_matrix):
            parents.append(ind)

    return parents

# precisa consertar
# Create the crossover operator
# Combine features of each solution using a crossover point


def rand_city_non_conditional():
    cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
              'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']
    index = random.randint(0, (len(cities) - 1))
    return cities[index]


def rand_city(seq):
    cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
              'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']
    available_cities = list(set(cities) - set(seq))

    if not available_cities:
        raise ValueError("Não há cidades disponíveis para seleção.")

    index = random.randint(0, len(available_cities) - 1)
    return available_cities[index]


def check_and_fix_path(distance_matrix, city_sequence):
    new_city_sequence = list(city_sequence)
    remaining_cities = list(
        set(distance_matrix.keys()) - set(new_city_sequence))

    current_city = rand_city_non_conditional()

    while remaining_cities:
        if current_city not in new_city_sequence:
            new_city_sequence.append(current_city)
            remaining_cities.remove(current_city)

        possible_cities = [
            city for city in remaining_cities if city in distance_matrix[current_city]]

        if not possible_cities:
            break

        next_city = rand_city_non_conditional()
        current_city = next_city

    if new_city_sequence[-1] != new_city_sequence[0] and new_city_sequence[-1] in distance_matrix.get(new_city_sequence[0], {}):
        new_city_sequence.append(new_city_sequence[0])

    return new_city_sequence


def crossover(parents, matriz):
    # random indexes to cross states with
    cross_points = random.sample(range(NUM_CITIES), MIXING_NUMBER - 1)
    offsprings = []
    # all permutations of parents
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))

    for perm in permutations:
        offspring = []

        # track starting index of sublist
        start_pt = 0

        # doesn't account for the last parent
        for parent_idx, cross_point in enumerate(cross_points):

            # sublist of parent to be crossed
            parent_part = perm[parent_idx][start_pt:cross_point]
            offspring += parent_part

            # update index pointer
            start_pt = cross_point

        # last parent
        last_parent = perm[-1]
        parent_part = last_parent[cross_point:]

        for city in parent_part:
            if city not in offspring:
                offspring.append(city)
        while len(offspring) < NUM_CITIES:
            new_city = rand_city(offspring)
            if new_city not in offspring:
                offspring.append(new_city)

    return offsprings


def complete_offspring(offspring):
    while len(offspring) < NUM_CITIES:
        new_city = rand_city(offspring)
        while new_city in offspring:
            new_city = rand_city(offspring)
        offspring.append(new_city)
    return offspring

# nao sei se precisa consertar
# Create the routine to mutate a solution
# A operator to create diversity in the population


def mutate(seq):
    for row in range(NUM_CITIES - 1):
        if random.random() < MUTATION_RATE:
            # Escolhe uma cidade aleatória do conjunto
            new_city = rand_city_non_conditional()
            if new_city not in seq:
                seq[row] = new_city
    return seq

# Print the solution


def print_found_goal(population, distance_matrix, generations, to_print=True):
    best_solution = None
    best_score = 0

    for ind in population:
        score = fitness_score(ind, distance_matrix)

        if to_print:
            print(f'{ind}. Score: {score}')

        if score > best_score and len(ind) == NUM_CITIES + 1:
            best_score = score
            best_solution = ind

    if generations == MAX_GEN:
        if to_print:
            print('Gerações máximas atingidas!')
            print(
                f'Melhor solução: {best_solution}, Score: {best_score}')
            return best_score
        return best_score

    return False
# Create the routine to implement the evolution


def evolution(population, distance_matrix, fitness_score):
    # select individuals to become parents
    parents = selection(population, distance_matrix)

    # recombination. Create new offsprings
    offsprings = crossover(parents, distance_matrix)

    # mutation
    offsprings = list(map(mutate, offsprings))

    # introduce top-scoring individuals from the previous generation and keep top fitness individuals
    new_gen = offsprings + population

    # Utilize a matriz de distâncias na avaliação de aptidão
    new_gen = sorted(new_gen, key=lambda ind: fitness_score(
        ind, distance_matrix), reverse=True)[:POPULATION_SIZE]

    return new_gen
# Running the experiment


generation = 0

# aparentemente ta bom, pra teste recomendo usar lista menor


def generate_population():
    population = []
    cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
              'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']

    # matriz de distancia entre cidades
    distance_matrix = {
        'João Monlevade': {'Ipatinga': 106, 'Belo Horizonte': 117, 'Capelinha': 359, 'Visc. Do Rio Branco': 213},
        'Ipatinga': {'Gov. Valadares': 106, 'João Monlevade': 106, 'Montes Claros': 525, 'Barbacena': 368},
        'Gov. Valadares': {'Capelinha': 214, 'Ipatinga': 106, 'Visc. Do Rio Branco': 372},
        'Capelinha': {'Montes Claros': 317, 'João Monlevade': 359, 'Gov. Valadares': 214, 'Araxá': 778},
        'Montes Claros': {'Belo Horizonte': 420, 'Capelinha': 317, 'Araxá': 566, 'Ipatinga': 525},
        'Belo Horizonte': {'João Monlevade': 117, 'Montes Claros': 420, 'Araxá': 362, 'Barbacena': 171},
        'Barbacena': {'Juiz de Fora': 101, 'Belo Horizonte': 171, 'Araxá': 506, 'Ipatinga': 368},
        'Juiz de Fora': {'Barbacena': 101, 'Visc. Do Rio Branco': 128},
        'Visc. Do Rio Branco': {'João Monlevade': 213, 'Juiz de Fora': 128, 'Gov. Valadares': 372},
        'Araxá': {'Belo Horizonte': 362, 'Barbacena': 506, 'Montes Claros': 566, 'Capelinha': 778}
    }

    for _ in range(POPULATION_SIZE):
        valid_route = []
        remaining_cities = cities.copy()
        current_city = random.choice(remaining_cities)

        while remaining_cities:
            if current_city not in valid_route:
                valid_route.append(current_city)
                remaining_cities.remove(current_city)

            possible_cities = [
                city for city in remaining_cities if city in distance_matrix[current_city]]

            if not possible_cities:
                break

            next_city = random.choice(possible_cities)
            current_city = next_city

        if valid_route[-1] != valid_route[0] and valid_route[-1] in distance_matrix.get(valid_route[0], {}):
            valid_route.append(valid_route[0])
            population.append(valid_route)

    return population, distance_matrix


def save_to_csv(generation, score, time, num_pop, mutation_rate, n_queens):
    with open('results_salesman.csv', mode='a', newline=f'\n') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['generation',
                            'score', 'time', 'num_pop', 'mutation_rate', 'n_queens'])
        writer.writerow([generation,
                        score, time, num_pop, mutation_rate, n_queens])


cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
          'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']

population, distance_matrix = generate_population()

initial_time = time.time()

# Generations until found the solution
while not print_found_goal(population, distance_matrix, generation):
    print(f'Generation: {generation}')
    print_found_goal(population, distance_matrix, generation)
    population = evolution(population, distance_matrix, fitness_score)
    generation += 1

end_time = time.time()
save_to_csv(generation, '?', round((end_time - initial_time), 2),
            POPULATION_SIZE, MUTATION_RATE, NUM_CITIES)

print('Tempo gasto: ', (end_time - initial_time))

# cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
#           'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']

# def generate_population(graph, cities):
#     population = []

# # Criar uma matriz de distâncias entre as cidades
# distance_matrix = {
#     'João Monlevade': {'Ipatinga': 106, 'Belo Horizonte': 117, 'Capelinha': 359, 'Visc. Do Rio Branco': 213},
#     'Ipatinga': {'Gov. Valadares': 106, 'João Monlevade': 106},
#     'Gov. Valadares': {'Capelinha': 214, 'Ipatinga': 106},
#     'Capelinha': {'Montes Claros': 317, 'João Monlevade': 359, 'Gov. Valadares': 214},
#     'Montes Claros': {'Belo Horizonte': 420, 'Capelinha': 317, 'Araxá': 566},
#     'Belo Horizonte': {'João Monlevade': 117, 'Montes Claros': 420, 'Araxá': 362, 'Barbacena': 117},
#     'Barbacena': {'Juiz de Fora': 101, 'Belo Horizonte': 117, 'Araxá': 506, },
#     'Juiz de Fora': {'Barbacena': 101, 'Visc. Do Rio Branco': 128},
#     'Visc. Do Rio Branco': {'João Monlevade': 213, 'Juiz de Fora': 128},
#     'Araxá': {'Belo Horizonte': 362, 'Barbacena': 506, 'Montes Claros': 566}

# }

#     for i in range(POPULATION_SIZE):
#         # Criar uma rota aleatória inicial
#         route = random.sample(cities, len(cities))
#         population.append(route)

#     return population, distance_matrix
