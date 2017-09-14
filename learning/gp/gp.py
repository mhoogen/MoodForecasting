import random
from eval.formal_method import EvaluationFramework
from model.model import Model
import math
from copy import deepcopy
from tree import Tree
from individual import Individual

class GP():

    states = []
    population = []
    population_size = 0
    generations = 0
    max_depth = 0
    training_data = []
    test_data = []
    max_params = 0
    parameter_names = []
    eval_aspects = []
    tournament_size = 4
    p_vector_crossover = 0.5
    max_tries = 3
    current_fitness_values = []
    best_fitness_values = []
    p_mutation = 0.5

    operators = {'*': {'output': 'numeric', 'arguments': ['numeric', 'numeric'], 'notation': 'infix'},
                 '+': {'output': 'numeric', 'arguments': ['numeric', 'numeric'], 'notation': 'infix'},
                 '-': {'output': 'numeric', 'arguments': ['numeric', 'numeric'], 'notation': 'infix'}}
            # ,
            # 'math.pow': {'output': 'numeric', 'arguments': ['numeric', 'numeric'], 'notation': 'prefix'}}
    terminals = {}

    def __init__(self):
        self.states = []
        self.population = []
        self.population_size = 0
        self.max_depth = 0
        self.max_params = 0
        self.parameter_names = []
        self.bestfitness_values = []
        self.max_tries = 3
        self.current_fitness_values = []
        self.best_fitness_values = []


    def initialize(self, states, population_size, max_depth, max_params, seed_used):
        self.states = states
        self.population_size = population_size
        self.max_depth = max_depth
        self.max_params = max_params
        self.parameter_names = []
        self.best_fitness_values = []
        self.current_fitness_values = []

        for s in states:
            self.terminals[s] = {'output':'numeric'}
        for p in range(max_params):
            par_name = 'self.param'+str(p)+'v'
            self.terminals[par_name] = {'output': 'numeric'}
            self.parameter_names.append(par_name)
        random.seed(seed_used)
        for i in range(self.population_size):
            ind = Individual()
            ind.initialize(self.states, self.operators, self.terminals, self.max_depth, self.parameter_names)
            self.population.append(ind)

    def determine_required_type(self, tree):
        if tree.is_terminal():
            return self.terminals[tree.operator]['output']
        else:
            return self.operators[tree.operator]['output']

    def mutate_tree(self, parent_individual, parent_tree):
        child = deepcopy(parent_tree)
        nodes = child.has_number_of_nodes()
        mutation_point = int(random.uniform(1, nodes+1))
        subtree = child.get_point(mutation_point)
        required_type = self.determine_required_type(subtree)
        subtree_depth = child.get_subtree_depth(subtree)
        new_subtree = parent_individual.create_individual_equation(subtree_depth, required_type)
        child.replace(mutation_point, new_subtree)
        return child

    def tree_level_crossover_tree(self, parent_individual, parent_tree1, parent_tree2):
        nodes_parent1 = parent_tree1.has_number_of_nodes()
        nodes_parent2 = parent_tree2.has_number_of_nodes()

        successful_crossover = False
        i = 0

        while not successful_crossover and i < self.max_tries:
            child = deepcopy(parent_tree1)
            mutation_point_parent1 = int(random.uniform(1, nodes_parent1+1))
            mutation_point_parent2 = int(random.uniform(1, nodes_parent2+1))
            subtree1 = child.get_point(mutation_point_parent1)
            subtree2 = parent_tree2.get_point(mutation_point_parent2)
            required_type_parent1 = self.determine_required_type(subtree1)
            required_type_parent2 = self.determine_required_type(subtree2)

            if required_type_parent1 == required_type_parent2:
                child.replace(mutation_point_parent1, subtree2)
                if child.has_depth() <= self.max_depth:
                    successful_crossover = True
                    return child
            i += 1
        return None

    def vector_level_crossover_tree(self, parent_tree1, parent_tree2):
        r = random.uniform(0, 1)
        if r > 0.5:
            return deepcopy(parent_tree1)
        else:
            return deepcopy(parent_tree2)

    def mutate(self, parent):
        child = deepcopy(parent)
        t_new = []
        i = (int) (random.uniform(0, 1) * len(parent.trees))
        for t in range(0, len(parent.trees)):
            if t == i:
                t_new.append(self.mutate_tree(parent, parent.trees[t]))
            else:
                t_new.append(parent.trees[t])
        child.trees = t_new
        return child

    def crossover(self, parent1, parent2):
        child = deepcopy(parent1)
        t_new = []
        for t in range(len(parent1.trees)):
            if random.uniform(0,1) > self.p_vector_crossover:
                new_tree = self.tree_level_crossover_tree(parent1, parent1.trees[t], parent2.trees[t])
                if new_tree == None:
                    new_tree = self.vector_level_crossover_tree(parent1.trees[t], parent2.trees[t])
                t_new.append(new_tree)
            else:
                t_new.append(self.vector_level_crossover_tree(parent1.trees[t], parent2.trees[t]))
        child.trees = t_new
        return child


    def mutation_prob(self, current_fitness_value, all_fitness_values):
        return 0.1 + 0.2 * (1-(current_fitness_value/float(max(all_fitness_values))))

    def tournament_selection(self, all_fitness_values, tournament_size):
        selected_individuals = []
        fitness_values_individuals = []
        for i in range(tournament_size):
            selected_index = int(random.uniform(0, len(all_fitness_values)))
            selected_individuals.append(selected_index)
            fitness_values_individuals.append(all_fitness_values[selected_index])
        return selected_individuals[fitness_values_individuals.index(max(fitness_values_individuals))]

    def get_population(self):
        model_population = []
        for individual in range(self.population_size):
            model_population.append(self.population[individual].return_as_model())
        return model_population

    def set_fitness(self, fitness_values):
        self.current_fitness_values = fitness_values

    def simplify(self, candidate):
        child = deepcopy(candidate)
        child.simpify_trees()
        return child


    def evolve_population(self):
        new_population = []

        # Elitist approach:

        reproductive_rate = 0.1
        best_fitness = max(self.current_fitness_values)
        print 'Best fitness: ' + str(best_fitness)
        self.best_fitness_values.append(best_fitness)
        best_individual = self.population[self.current_fitness_values.index(best_fitness)]
        print best_individual.return_as_model().print_model()

        new_population.append(best_individual)

        while len(new_population) < self.population_size:
            index_parent1 = self.tournament_selection(self.current_fitness_values, self.tournament_size)
            pm = self.mutation_prob(self.current_fitness_values[index_parent1], self.current_fitness_values)
            r = random.uniform(0, 1)
            # crossover
            if r > pm and r <= (1-reproductive_rate):
                index_parent2 = self.tournament_selection(self.current_fitness_values, self.tournament_size)
                parent1 = self.population[index_parent1]
                parent2 = self.population[index_parent2]
                child = self.crossover(parent1, parent2)
                new_population.append(self.simplify(child))
            # mutation
            elif r <= pm:
                parent = self.population[index_parent1]
                child = self.mutate(parent)
                new_population.append(self.simplify(child))
            # add the original parent.
            else:
                new_population.append(self.population[index_parent1])

        self.population = new_population
        return best_individual.return_as_model()
