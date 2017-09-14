import random
import math
from copy import deepcopy

class GA():

    population = []
    population_size = 0
    number_of_states = 0
    p_c = 0
    p_m = 0
    current_fitness_values = []
    best_fitness_values = []
    tournament_size = 2

    def __init__(self):
        self.population = []
        self.population_size = 0
        self.number_of_states = 0
        self.p_c = 0
        self.p_m = 0
        self.current_fitness_values = []
        self.best_fitness_values = []


    def initialize(self, number_of_states, population_size, p_m, p_c, seed_used):
        self.number_of_states = number_of_states
        self.population_size = population_size
        self.p_m = p_m
        self.p_c = p_c
        random.seed(seed_used)
        self.population = []
        self.current_fitness_values = []
        self.best_fitness_values = []

        for p in range(0, population_size):
            individual = []
            for s in range(0, number_of_states):
                if random.uniform(0, 1) > 0.5:
                    individual.append(1)
                else:
                    individual.append(0)
            self.population.append(individual)

    def mutate(self, parent):
        child = deepcopy(parent)
        for i in range(0, len(child)):
            if random.uniform(0, 1) <= self.p_m:
                child[i] = 1-child[i]
        return child

    def crossover(self, parent1, parent2):
        crossover_point = int(random.uniform(0, self.number_of_states))
        child = parent1[0:crossover_point] + parent2[crossover_point:self.number_of_states]
        return child

    def routlette_wheel_selection(self, all_fitness_values, tournament_size):
        selected_individuals = []
        fitness_values_individuals = []
        for i in range(tournament_size):
            selected_index = int(random.uniform(0, len(all_fitness_values)))
            selected_individuals.append(selected_index)
            fitness_values_individuals.append(all_fitness_values[selected_index])
        return selected_individuals[fitness_values_individuals.index(max(fitness_values_individuals))]

    def get_population(self):
        return self.population

    def set_fitness(self, fitness_values):
        self.current_fitness_values = fitness_values

    def evolve_population(self):
        new_population = []

        # Elitist approach:

        best_fitness = max(self.current_fitness_values)
        self.best_fitness_values.append(best_fitness)
        best_individual = self.population[self.current_fitness_values.index(best_fitness)]

        new_population.append(best_individual)

        while len(new_population) < self.population_size:
            index_parent1 = self.routlette_wheel_selection(self.current_fitness_values, self.tournament_size)
            parent1 = self.population[index_parent1]
            if random.uniform(0, 1) <= self.p_c:
                index_parent2 = self.routlette_wheel_selection(self.current_fitness_values, self.tournament_size)
                parent2 = self.population[index_parent2]
                child = self.crossover(parent1, parent2)
            else:
                child = parent1
            mutated_child = self.mutate(child)
            new_population.append(mutated_child)
        self.population = new_population
        return best_individual
