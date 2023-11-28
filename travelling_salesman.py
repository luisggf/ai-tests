# Constants, experiment parameters
import itertools
from scipy import special as sc
import time
from weighted_graph import *
import csv
import random
NUM_CITIES = 4
POPULATION_SIZE = 5
MIXING_NUMBER = 4
MUTATION_RATE = 0.05

# ta puro


def fitness_score(seq, distance_matrix):
    total_distance = 0

    try:
        # Verifica se a sequência é inválida
        if seq[0] != seq[-1]:
            return None

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
                return None

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


def crossover(parents):
    MIXING_NUMBER = len(parents)

    # O número de pontos de cruzamento não deve ser maior que o número de cidades
    num_crossover_points = min(NUM_CITIES - 1, MIXING_NUMBER - 1)
    cross_points = random.sample(range(NUM_CITIES), num_crossover_points)
    offsprings = []

    # All permutations of parents
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))

    for perm in permutations:
        offspring = []
        start_pt = 0

        for parent_idx, cross_point in enumerate(cross_points):
            parent_part = perm[parent_idx][start_pt:cross_point]
            offspring.append(parent_part)
            start_pt = cross_point

        last_parent = perm[-1]
        parent_part = last_parent[start_pt:]
        offspring.append(parent_part)

        flat_offspring = list(itertools.chain(*offspring))
        unique_offspring = list(dict.fromkeys(flat_offspring))

        if len(unique_offspring) == NUM_CITIES and unique_offspring[0] == unique_offspring[-1]:
            offsprings.append(unique_offspring)

    return offsprings


# funcao original
# def crossover(parents):

#     # random indexes to to cross states with
#     cross_points = random.sample(range(NUM_CITIES), MIXING_NUMBER - 1)
#     offsprings = []

#     # all permutations of parents
#     permutations = list(itertools.permutations(parents, MIXING_NUMBER))

#     for perm in permutations:
#         offspring = []

#         # track starting index of sublist
#         start_pt = 0

#         # doesn't account for last parent
#         for parent_idx, cross_point in enumerate(cross_points):

#             # sublist of parent to be crossed
#             parent_part = perm[parent_idx][start_pt:cross_point]
#             offspring.append(parent_part)

#             # update index pointer
#             start_pt = cross_point

#         # last parent
#         last_parent = perm[-1]
#         parent_part = last_parent[cross_point:]
#         offspring.append(parent_part)

#         # flatten the list since append works kinda differently
#         offsprings.append(list(itertools.chain(*offspring)))

#     return offsprings


# nao sei se precisa consertar
# Create the routine to mutate a solution
# A operator to create diversity in the population
def mutate(seq, cities):
    for row in range(NUM_CITIES - 1):
        if random.random() < MUTATION_RATE:
            # Escolhe uma cidade aleatória do conjunto
            new_city = random.choice(cities)
            if new_city not in seq:
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

# aparentemente ta bom, pra teste recomendo usar lista menor


def generate_population():
    population = []
    cities = ['Ouro Preto', 'Mariana', 'Alvinopolis', 'BH']

    distance_matrix = {
        'Ouro Preto': {'Mariana': 14.4, 'BH': 101.5},
        'Mariana': {'Ouro Preto': 14.4, 'Alvinopolis': 69.3},
        'Alvinopolis': {'Mariana': 69.3, 'BH': 162},
        'BH': {'Ouro Preto': 69.3, 'Alvinopolis': 162}
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


def save_to_csv(generation, score, time, num_pop, mutation_rate, n_queens):
    with open('results_salesman.csv', mode='a', newline=f'\n') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['generation',
                            'score', 'time', 'num_pop', 'mutation_rate', 'n_queens'])
        writer.writerow([generation,
                        score, time, num_pop, mutation_rate, n_queens])


cities_graph = Weighted_Graph()

# cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
#           'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']

cities = ['Ouro Preto', 'Mariana', 'Alvinopolis', 'BH']

# Generate Random Population
population, distance_matrix = generate_population()

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
