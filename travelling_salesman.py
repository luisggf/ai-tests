import itertools
from scipy import special as sc
import time
from weighted_graph import *
from itertools import chain
import csv
import random
NUM_CITIES = 10
MAX_GEN = 100
POPULATION_SIZE = 10
MIXING_NUMBER = 9
MUTATION_RATE = 0.05


def fitness_score(seq, distance_matrix):
    total_distance = 0

    try:
        while True:
            # reinicializa a distancia acumulada para cada tentativa
            total_distance = 0

            # calcula a distância acumulada entre nós na sequência dada
            for i in range(len(seq) - 1):
                city1 = seq[i]
                city2 = seq[i + 1]

                # verifica se ha um caminho entre as cidades na matriz de distâncias
                if city2 in distance_matrix.get(city1, {}):
                    distance_between_cities = distance_matrix[city1][city2]
                    total_distance += distance_between_cities
                else:
                    # se não houver caminho, corrige a sequencia e tenta denovo
                    seq = check_and_fix_path(distance_matrix, seq)
                    break
            # se percorreu toda a sequência sem problemas, encerra o loop
            else:
                break

        return total_distance

    except Exception as error:
        print(error)


# Create the selection operator acording their fitness score
# Select best solutions for next step: crossover

def selection(population, distance_matrix):
    # lista para armazenar os indivíduos selecionados como pais
    parents = []

    # variável criada para armazenar a soma total dos inversos do fitness
    total_inverse_fitness = 0

    # calcula os inversos do fitness para cada indivíduo na população
    inverse_fitness_values = [
        1 / fitness_score(ind, distance_matrix) for ind in population]

    # calcula a soma total dos inversos do fitness
    total_inverse_fitness = sum(inverse_fitness_values)

    # calcula as probabilidades de seleção para cada indivíduo
    selection_probabilities = [
        inv_fitness / total_inverse_fitness for inv_fitness in inverse_fitness_values]

    # realiza a seleção proporcional com base nas probabilidades calculadas
    for _ in range(2):
        # seleciona um índice com base nas probabilidades usando random.choices
        selected_index = random.choices(
            range(len(population)), weights=selection_probabilities)[0]
        # adiciona o indivíduo selecionado na lista de pais
        parents.append(population[selected_index])
        print(len(parents))

    return parents


# Create the crossover operator
# Combine features of each solution using a crossover point

# retorna uma cidade aleatória
def rand_city_non_conditional():
    cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
              'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']
    index = random.randint(0, (len(cities) - 1))
    return cities[index]

# retorna uma cidade aleatória baseada em um conjunto ja existente


def rand_city(seq):
    cities = ['Montes Claros', 'Capelinha', 'Gov. Valadares', 'Ipatinga', 'João Monlevade',
              'Visc. Do Rio Branco', 'Juiz de Fora', 'Barbacena', 'Belo Horizonte', 'Araxá']
    available_cities = list(set(cities) - set(seq))

    if not available_cities:
        raise ValueError("Não há cidades disponíveis para seleção.")

    index = random.randint(0, len(available_cities) - 1)
    return available_cities[index]

# corrige o caminho da sequencia passada por parametro caso seja invalido e retorna uma sequencia valida


def remove_duplicatas_mais_antigas(lista):
    elementos_vistos = []
    lista_sem_duplicatas = []

    for elemento in lista:
        # Verifica se o elemento já foi visto
        if elemento not in elementos_vistos:
            # Adiciona o elemento único à lista sem duplicatas
            lista_sem_duplicatas.append(elemento)

            # Adiciona o elemento à lista de elementos vistos
            elementos_vistos.append(elemento)

    return lista_sem_duplicatas


def check_and_fix_path(distance_matrix, city_sequence):
    new_city_sequence = set(city_sequence)
    remaining_cities = list(
        set(distance_matrix.keys()) - set(new_city_sequence))
    if not remaining_cities:
        return new_city_sequence

    new_city_sequence = list(new_city_sequence)

    while remaining_cities:
        possible_cities = [
            city for city in remaining_cities if city in distance_matrix[new_city_sequence[-1]]
        ]

        if not possible_cities:
            break

        next_city = random.choice(possible_cities)
        new_city_sequence.append(next_city)
        remaining_cities.remove(next_city)

    # Se a última cidade estiver na lista de adjascencia da primeira e o tamanho da sequência for o correto
    if new_city_sequence[-1] in distance_matrix.get(new_city_sequence[0], {}) and len(new_city_sequence) == NUM_CITIES:
        new_city_sequence.append(new_city_sequence[0])

    return new_city_sequence


def crossover(parents, matriz):
    # random indexes to cross states with
    cross_points = random.sample(range(NUM_CITIES), 1)
    offsprings = []
    parent = parents[0]
    last_parent = parents[-1]

    if parent == last_parent:
        return parent

    # Selecting subsequences from the two parents
    offspring = []
    start_pt = 0
    for _, cross_point in enumerate(cross_points):
        parent_part = parent[start_pt:cross_point]
        offspring += parent_part
        start_pt = cross_point

    # Handling the last parent
    offspring += last_parent[start_pt:]
    offspring = check_and_fix_path(matriz, offspring)
    offsprings.append(offspring)

    return offsprings

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
    best_score = float('inf')

    for ind in population:
        score = fitness_score(ind, distance_matrix)

        if to_print:
            print(f'{ind}. Score: {score}')

        if score < best_score and len(ind) == NUM_CITIES + 1:
            best_score = score
            best_solution = ind

    if generations == MAX_GEN:
        if to_print:
            print('Gerações máximas atingidas!')
            print(
                f'Melhor solução encontrada na geração {generations}: {best_solution}. Score: {best_score}')
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
    # offsprings = list(map(mutate, offsprings))

    # introduce top-scoring individuals from the previous generation and keep top fitness individuals
    new_gen = offsprings + population

    # Utilize a matriz de distâncias na avaliação de aptidão
    # new_gen = sorted(new_gen, key=lambda ind: fitness_score(
    #     ind, distance_matrix), reverse=True)[:POPULATION_SIZE]

    return new_gen
# Running the experiment


def generate_population(start_option, cities, distance_matrix):
    population = []
    valid_route = []
    num_pop = 0
    while num_pop < POPULATION_SIZE:
        valid_route = []
        valid_route.append(start_option)
        remaining_cities = cities.copy()
        remaining_cities.remove(start_option)

        while remaining_cities:
            current_city = valid_route[-1]

            possible_cities = [
                city for city in remaining_cities if city in distance_matrix[current_city]
            ]

            if not possible_cities:
                break

            next_city = random.choice(possible_cities)
            valid_route.append(next_city)
            remaining_cities.remove(next_city)

        if valid_route[-1] in distance_matrix.get(start_option, {}) and len(valid_route) == NUM_CITIES:
            valid_route.append(valid_route[0])
            population.append(valid_route)
            num_pop += 1

    return population, distance_matrix


def save_to_csv(generation, score, time, num_pop, mutation_rate, n_queens):
    with open('results_salesman.csv', mode='a', newline=f'\n') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['generation',
                            'score', 'time', 'num_pop', 'mutation_rate', 'n_cities'])
        writer.writerow([generation,
                        score, time, num_pop, mutation_rate, n_queens])


start_option = rand_city_non_conditional()
generation = 0

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

population, distance_matrix = generate_population(
    start_option, cities, distance_matrix)
initial_time = time.time()

# Generations until found the solution
while not print_found_goal(population, distance_matrix, generation):
    print(f'Generation: {generation}')
    score = print_found_goal(population, distance_matrix, generation)
    population = evolution(population, distance_matrix, fitness_score)
    generation += 1

end_time = time.time()

save_to_csv(generation, score, round((end_time - initial_time), 2),
            POPULATION_SIZE, MUTATION_RATE, NUM_CITIES)

print('Tempo gasto: ', (end_time - initial_time))
