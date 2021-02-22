import random
from typing import List, TypedDict

import django
import numpy as np
import pygad  # type: ignore

from .models import Course


class Scheduling:
    def __init__(self):
        self.int_to_subject_map = {}
        self.subject_to_int_map = {}
        self.subject_to_year_map = {}
        self.num_subj = 0
        self.num_slots = 0

    def randomize_initialization(self):
        lis = [i for i in self.int_to_subject_map.keys()]
        random.shuffle(lis)
        array = np.empty((20, self.num_subj), dtype="object")
        for i in range(20):
            array[i] = lis
            random.shuffle(lis)

        return array

    def find_repeated_conflicts(self, array: np.array):
        lis = {i: 0 for i in self.int_to_subject_map.keys()}
        for i in array:
            lis[i] += 1
        count = 0
        for i in self.int_to_subject_map.keys():
            if lis[i] >= 1:
                count += lis[i] - 1
        print("repeated:" + str(count))
        return count

    def find_same_year_conflicts(self, array: np.array):
        count = 0
        for i in range(int(self.num_subj / self.num_slots)):
            sample = array[i * self.num_slots : i * self.num_slots + self.num_slots]
            years = [
                self.subject_to_year_map[self.int_to_subject_map[i]] for i in sample
            ]
            uniques = np.unique(years)
            for i in uniques:
                count += years.count(i) - 1
        print("same_year:" + str(count))
        return count

    def count_conflicts(self, array: np.array):
        count = self.find_repeated_conflicts(array)
        count1 = self.find_same_year_conflicts(array)

        return count + count1


mapping_variables = Scheduling()
best_solution = 0
best_fitness = 0


def on_gen(ga_instance):
    global best_fitness, best_solution

    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness > best_fitness:
        best_fitness = solution_fitness
        best_solution = solution

    if (mapping_variables.find_repeated_conflicts(solution) == 0) and (
        mapping_variables.find_same_year_conflicts(solution) == 0
    ):
        print("ending")
        return "stop"


def fitness_func(solution: np.array, solution_idx: int):
    fitness = mapping_variables.num_subj - mapping_variables.count_conflicts(solution)
    return fitness


def get_res(
    num_iterations: int,
    num_slots: int,
    courses: List[TypedDict("Course", {"code": str, "semester": int})],
):
    mapping_variables.num_slots = num_slots
    mapping_variables.subject_to_year_map = {
        course["code"]: course["semester"] for course in courses
    }
    mapping_variables.num_subj = len(mapping_variables.subject_to_year_map)

    count = 1
    for course_code in mapping_variables.subject_to_year_map.keys():
        mapping_variables.subject_to_int_map[course_code] = count
        mapping_variables.int_to_subject_map[count] = course_code
        count += 1

    print(
        mapping_variables.subject_to_int_map,
        mapping_variables.int_to_subject_map,
        mapping_variables.subject_to_year_map,
    )

    x = mapping_variables.randomize_initialization()
    count1 = 0
    ga_instance = pygad.GA(
        num_generations=num_iterations,
        num_parents_mating=8,
        initial_population=x,
        fitness_func=fitness_func,
        parent_selection_type="rank",
        keep_parents=8,
        crossover_type="uniform",
        mutation_type="swap",
        on_generation=on_gen,
    )
    ga_instance.run()

    solution_values = [
        mapping_variables.int_to_subject_map[i] for i in best_solution.tolist()
    ]
    print(solution_values)

    return solution_values


# college, department, num_slot, even/odd
