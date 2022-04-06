import sys
import random
from enum import Enum, auto

import plotly.express


class Crossover(Enum):
    PMX = auto()
    CX = auto()
    OX = auto()


class Mutation(Enum):
    REPLACEMENT = auto()
    INVERSION = auto()


class Selection(Enum):
    TOURNAMENT = auto()
    PROPORTIONAL = auto()


def load_file(file_path: str) -> list:
    f = open(file_path)
    lines = f.readlines()
    f.close()
    return lines


def get_distance_matrix(lines: list) -> list:
    matrix = [[0 for _ in range(int(lines[0]))] for _ in range(int(lines[0]))]
    i = 0
    for line in lines[1:]:
        line = line.strip().split(' ')
        j = 0
        for dist in line:
            matrix[i][j] = int(dist)
            matrix[j][i] = int(dist)
            j += 1
        i += 1

    return matrix


# n random list of m indexes
def generate_population(n: int, m: int) -> list:
    return [random.sample(range(m), m) for _ in range(n)]


def fitness(cost_array: list, solution: list) -> int:
    mark = sum([cost_array[solution[index]][solution[index + 1]] for index in
                range(len(solution) - 1)])
    return mark + cost_array[solution[-1]][solution[0]]


# returns index of the minimum mark of given indexes
def best_index(marks, indexes):
    t = [marks[i] for i in indexes]
    min_mark = min(t)
    min_index = indexes[t.index(min_mark)]
    return min_index


# Selekcja turniejowa
def selection_turn(population, marks, k=3):
    return [population[
                best_index(marks, random.sample(range(len(population)), k))
            ] for _ in population]


# Selekcja koła ruletki
def selection_prop(population, marks):
    return random.choices(population, weights=[
        max(marks) + 1 - mark for mark in marks
    ], k=len(population))


def selection(selection: Selection, population: list, marks: list, k: int = 3):
    return selection_turn(population, marks, k) if \
        selection == Selection.TOURNAMENT else selection_prop(population,
                                                              marks)


def crossing_pmx(p1, p2):
    [x1, x2] = random.sample(range(len(p1)), 2)
    if x1 > x2:
        x1, x2 = x2, x1

    c2 = [i for i in p1]
    c1 = [i for i in p2]

    _map1 = {}
    _map2 = {}

    for i in range(x1, x2):
        c2[i] = p2[i]
        _map1[p2[i]] = p1[i]
        c1[i] = p1[i]
        _map2[p1[i]] = p2[i]

    for i in list(range(x1)) + list(range(x2, len(p1))):
        while c2[i] in _map1:
            c2[i] = _map1[c2[i]]

        while c1[i] in _map2:
            c1[i] = _map2[c1[i]]

    return c1, c2


def crossing_ox(p1, p2):
    [x1, x2] = random.sample(range(len(p1)), 2)
    if x1 > x2:
        x1, x2 = x2, x1

    c1 = [-9999999 for _ in p1]
    c2 = [-9999999 for _ in p2]

    for i in range(x1, x2):
        c1[i] = p1[i]
        c2[i] = p2[i]

    c1_i = [i for i in list(range(len(p1))) if i not in list(range(x1, x2))]
    c2_i = [i for i in list(range(len(p1))) if i not in list(range(x1, x2))]
    for i in list(range(x2, len(p1))) + list(range(x2)):
        if p2[i] not in c1:
            c1[c1_i.pop(0)] = p2[i]
        if p1[i] not in c2:
            c2[c2_i.pop(0)] = p1[i]

    return c1, c2


def crossing_cx(p1: list, p2: list):
    cycle = []
    cycle_index = []

    for i, val in enumerate(p1):
        if val == p2[i]:
            continue

        if val in cycle:
            break

        if len(cycle) == 0:
            cycle.append(val)
            cycle_index.append(i)
            break

    if len(cycle) == 0:
        return p1, p2

    t_val = cycle[0]
    while True:
        i = p2.index(t_val)

        t_val = p1[i]
        t_index = p1.index(t_val)

        if t_val in cycle:
            break

        cycle.append(t_val)
        cycle_index.append(t_index)

    c1 = [p1[i] if i in cycle_index else p2[i] for i in range(len(p1))]
    c2 = [p2[i] if i in cycle_index else p1[i] for i in range(len(p1))]

    return c1, c2


def crossover(population: list, alg: Crossover,
              pk: float = 0.95) -> list:
    new_population = []
    for i in range(0, len(population), 2):
        new_1, new_2 = population[i][:], population[i + 1][:]
        if random.random() < pk and new_1 != new_2:
            new_1, new_2 = crossing_ox(
                population[i],
                population[i + 1]) if alg == Crossover.OX else \
                crossing_cx(
                    population[i],
                    population[i + 1]) if alg == Crossover.CX else \
                    crossing_pmx(population[i],
                                 population[i + 1])
        new_population.extend([new_1, new_2])

    return new_population


# mutacja przez zamiane
def mutate_replace(solution: list, mutation_rate: float = 0.01):
    for i in range(len(solution)):
        if random.random() <= mutation_rate:
            j = random.randrange(len(solution))
            solution[i], solution[j] = solution[j], solution[i]


# mutacja przez inwersję
def mutate_inversion(solution: list, mutation_rate: float = 0.01):
    if random.random() <= mutation_rate:
        [i, j] = random.sample(range(len(solution)), 2)

        if i > j:
            i, j = j, i

        solution[i:j] = solution[i:j][::-1]


def mutate(population, mutation: Mutation, mutation_rate: float = 0.05):
    for solution in population:
        mutate_inversion(solution,
                         mutation_rate) if mutation == Mutation.INVERSION \
            else mutate_replace(solution, mutation_rate)


def loader_point(total, iteration):
    i = iteration + 1
    point = total / 100
    increment = total / 100
    if i % (1 * point) == 0:
        sys.stdout.write("\r[" + "=" * int(i / increment) + " " * int(
            ((total - i) / increment)) + "] " + str(
            int(i / total * 100)) + "%")
        sys.stdout.flush()


def tsp(
        file_path: str,
        population_size: int = 400,
        generations: int = 3000,
        mutation_prob: float = 0.05,
        crossover_prob: float = 0.7,
        selection_type: Selection = Selection.TOURNAMENT,
        crossover_type: Crossover = Crossover.PMX,
        k: int = 3,
        loader: bool = False,
        modify: bool = False
):
    city_list = get_distance_matrix(load_file(file_path))

    actual_population = generate_population(population_size, len(city_list))

    marks = [fitness(city_list, solution) for solution in
             actual_population]

    min_distance = min(marks)
    min_distance_index = marks.index(min_distance)
    min_solution = actual_population[min_distance_index]

    results = []

    for generation in range(generations):
        if loader:
            loader_point(generations, generation, )
        tmp_gen = selection(
            selection_type,
            population=actual_population,
            marks=marks,
            k=k
        )

        tmp_gen = crossover(tmp_gen, crossover_type, crossover_prob)

        # if generation / generations > 0.85:
        if modify and len(results) > 2000/20 and results[-50] == results[-1]:
            mutation_prob = 0.04
            crossover_prob = 0.95
            crossover_type = Crossover.CX

        if generation % 3 == 0:
            mutate(tmp_gen, Mutation.REPLACEMENT, mutation_prob)
        else:
            mutate(tmp_gen, Mutation.INVERSION, mutation_prob)

        marks = [fitness(city_list, solution) for solution in tmp_gen]

        tmp_min_distance = min(marks)

        if tmp_min_distance < min_distance:
            min_distance, min_distance_index = tmp_min_distance, \
                                               marks.index(tmp_min_distance)
            min_solution = actual_population[min_distance_index]

        actual_population = tmp_gen

        if generation % 20 == 0:
            results.append(tmp_min_distance)

    print()

    return min_solution, min_distance, results

    # mutacja przez zamianę jest gorsza


def show_result(solution, mark, show_plot=False, results=None):
    if results is None:
        results = []

    print("-".join([str(i) for i in solution]), mark)

    if show_plot:
        fig = plotly.express.line(x=[i*20 for i in range(len(results))], y=results)
        fig.write_html('result.html')


def test_fitness():
    l = [int(i) for i in
         "1-6-41-29-22-20-16-2-17-30-21-0-48-31-44-18-40-7-8-9-42-32-50-10-51-13-12-46-25-26-27-11-24-3-5-14-4-37-39-38-35-34-33-36-23-47-45-43-15-49-19-28".split(
             '-')]

    print(fitness(get_distance_matrix(load_file('berlin52.txt')), l))


def main():
    (solution, mark, results) = tsp("berlin52.txt",
                                    generations=10000,
                                    population_size=500,
                                    crossover_type=Crossover.PMX,
                                    mutation_prob=0.043,
                                    crossover_prob=.91,
                                    selection_type=Selection.PROPORTIONAL,
                                    k=2,
                                    loader=True)

    show_result(solution, mark, True, results)


# na początku wysoka wartość krzyżowania, potem zminiejszyć (albo wyłączyć)
if __name__ == '__main__':
    main()

    # 1-6-41-29-22-20-16-2-17-30-21-0-48-31-44-18-40-7-8-9-42-32-50-10-51-13-12-46-25-26-27-11-24-3-5-14-4-37-39-38-35-34-33-36-23-47-45-43-15-49-19-28 7938
    # 45-37-47-39-23-38-4-44-14-18-5-9-3-8-24-7-11-40-27-2-26-16-25-20-46-41-12-6-13-1-51-29-10-49-50-19-32-22-42-30-35-17-34-21-33-0-43-31-15-48-28-36 8193
