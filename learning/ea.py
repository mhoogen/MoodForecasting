from gp.gp import GP
from ga.ga import GA
from eval.formal_method import EvaluationFramework
from copy import deepcopy
import multiprocessing as mp
from util.util import Util
import pandas as pd
import numpy as np
import datetime
import dispy
import time, socket

def compute(n):
    print n
    host = socket.gethostname()
    return (host, n)

class EA():

    training_data = []
    test_data = []
    eval_aspects = []
    output = mp.Queue()
    util = Util()
    output_directory = ''
    parallel = False
    processors = 1
    cache_fitness = {}

    def create_model_coev(self, model, attributes):
        new_model = deepcopy(model)
        extra_parameter = 0
        model_attribute_number = 0

        for eq_number in range(0, len(new_model.state_names)):
            if not new_model.state_names[eq_number] in self.eval_aspects:
                if attributes[model_attribute_number] == 0:
                    if not new_model.state_equations[eq_number] in new_model.parameter_names:
                        param_name = 'self.substituted_value_' + str(extra_parameter)
                        new_model.state_equations[eq_number] = param_name
                        new_model.parameter_names.append(param_name)
                        extra_parameter += 1
                model_attribute_number += 1
        return new_model

    def in_cache(self, model):
        m = model.to_string()
        if m in self.cache_fitness.keys():
            return self.cache_fitness[m]
        return -1

    def evaluate_individual_gp(self, model, gp_index, cache=True, pop_size=3, generations=10):
        if cache:
            fitness_value = self.in_cache(model)
        else:
            fitness_value = -1
        if fitness_value == -1:
            ef = EvaluationFramework()
            fitness_value = ef.evaluate_model(model, self.training_data, self.test_data, self.eval_aspects, pop_size=3, generations=10)
        fitness_record = [gp_index, fitness_value]
        self.output.put(fitness_record)
        return fitness_record

    def evaluate_individual_coev(self, model, gp_index, ga_index):
        ef = EvaluationFramework()
        fitness_value = ef.evaluate_model(model, self.training_data, self.test_data, self.eval_aspects)
        fitness_record = [gp_index, ga_index, fitness_value]
        self.output.put(fitness_record)
        return fitness_record

#    def evaluate_population_gp(self, population_gp, cache=True, pop_size=5, generations=5):
#        self.output = mp.Queue()
#        fitness_values_gp = [0] * len(population_gp)
#        processes = []
#        results = []
#
#        for i in range(0, len(population_gp)):
#            m = population_gp[i]
#            if self.parallel:
#                processes.append(mp.Process(target=self.evaluate_individual_gp, args=(m, i, cache, pop_size, generations)))
#            else:
#                temp = self.evaluate_individual_gp(m, i, cache, pop_size, generations)
#                results.append(temp)
#
#        if self.parallel:
#            # start the processes given the number of processors
#            processes_completed = False
#            start = 0
#            while not processes_completed:
#                for p in range(start, min(start+self.processors, len(processes))):
#                    processes[p].start()
#                for p in range(start, min(start+self.processors, len(processes))):
#                    processes[p].join()
#                start += self.processors
#                if start >= len(processes):
#                    processes_completed = True
#
#            results = [self.output.get() for p in processes]
#
#        for result in results:
#            gp_index = result[0]
#            f = result[1]
#            # print 'gp_index is now: ', gp_index
#            # print 'f is now: ', f
#            if not (population_gp[gp_index].to_string() in self.cache_fitness.keys()):
#                self.cache_fitness[population_gp[gp_index].to_string()] = f
#                fitness_values_gp[gp_index] = f
#            else:
#                fitness_values_gp[gp_index] = self.cache_fitness[population_gp[gp_index].to_string()]
#
#        return fitness_values_gp

#    def compute(self, args):
#        result = self.evaluate_individual_gp(args[0], args[1], args[2], args[3], args[4])
#        host = socket.gethostname()
#        return (host, result)

    def evaluate_population_gp(self, population_gp, cache=True, pop_size=5, generations=5):
        if self.parallel:
            cluster = dispy.JobCluster(compute)
            jobs = []

        self.output = mp.Queue()
        fitness_values_gp = [0] * len(population_gp)
        processes = []
        results = []

        for i in range(0, len(population_gp)):
            m = population_gp[i]
            if self.parallel:
                job = cluster.submit([m, i, cache, pop_size, generations])
                job.id = i
                jobs.append(job)
            else:
                temp = self.evaluate_individual_gp(m, i, cache, pop_size, generations)
                results.append(temp)

        if self.parallel:
            cluster.wait()
            # start the processes given the number of processors
            for job in jobs:
                host, result = job()
                print result
                results.append(result)
            cluster.print_status()

        for result in results:
            gp_index = result[0]
            f = result[1]
            # print 'gp_index is now: ', gp_index
            # print 'f is now: ', f
            if not (population_gp[gp_index].to_string() in self.cache_fitness.keys()):
                self.cache_fitness[population_gp[gp_index].to_string()] = f
                fitness_values_gp[gp_index] = f
            else:
                fitness_values_gp[gp_index] = self.cache_fitness[population_gp[gp_index].to_string()]

        return fitness_values_gp



    def evaluate_populations_coev(self, population_gp, population_ga):
        self.output = mp.Queue()
        fitness_values_gp = [0] * len(population_gp)
        fitness_values_ga = [0] * len(population_ga)
        processes = []
        results = []

        for i in range(0, len(population_ga)):
            for j in range(0, len(population_gp)):
                m = self.create_model_coev(population_gp[j], population_ga[i])
                if self.parallel:
                    processes.append(mp.Process(target=self.evaluate_individual_coev, args=(m, j, i)))
                else:
                    results.append(self.evaluate_individual(m, j, i))

        if self.parallel:
            # start the processes given the number of processors
            processes_completed = False
            start = 0
            while not processes_completed:
                for p in range(start, min(start+self.processors, len(processes))):
                    processes[p].start()
                for p in range(start, min(start+self.processors, len(processes))):
                    processes[p].join()
                    print 'eval number ' + str(p)
                start += self.processors
                if start >= len(processes):
                    processes_completed = True

            results = [self.output.get() for p in processes]

        for result in results:
            gp_index = result[0]
            ga_index = result[1]
            f = result[2]
            fitness_values_gp[gp_index] = max(fitness_values_gp[gp_index], f)
            fitness_values_ga[ga_index] = max(fitness_values_gp[ga_index], f)
        return fitness_values_ga, fitness_values_gp

    def run_coev(self, states, population_size_gp, population_size_ga, generations, max_model_depth,
               training_data, test_data, max_params, eval_aspects, mutation_p_ga, crossover_p_ga, parallel=False, processors=1):
        self.training_data = training_data
        self.test_data = test_data
        self.eval_aspects = eval_aspects
        self.output_directory = self.util.create_output_directory()
        self.parallel = parallel
        self.processors = processors

        # Create the initial population of models in the GP

        gp_algorithm = GP()
        gp_algorithm.initialize(states, population_size_gp, max_model_depth, max_params, 0)
        ga_algorithm = GA()
        ga_algorithm.initialize(len(states)-len(eval_aspects), population_size_ga, mutation_p_ga, crossover_p_ga, 0)

        for i in range(0, generations):
            print 'generation number: ' + str(i)
            model_population = gp_algorithm.get_population()
            feature_population = ga_algorithm.get_population()
            fitness_values_ga, fitness_values_gp = self.evaluate_populations_coev(model_population, feature_population)
            gp_algorithm.set_fitness(fitness_values_gp)
            ga_algorithm.set_fitness(fitness_values_ga)
            best_individual_gp = gp_algorithm.evolve_population()
            best_individual_ga = ga_algorithm.evolve_population()
            self.util.write_results_to_file(self.output_directory, fitness_values_gp, fitness_values_ga, self.create_model(best_individual_gp, best_individual_ga), i)

    def evaluate_params_nsga_2(self, states, population_size_gp, generations, max_model_depth,
               training_data, test_data, max_params, eval_aspects, parallel=False, processors=1):
        # Generate a random population first.
        self.training_data = training_data
        self.test_data = test_data
        self.eval_aspects = eval_aspects
        self.parallel = parallel
        self.processors = processors

        gp_algorithm = GP()
        gp_algorithm.initialize(states, population_size_gp, max_model_depth, max_params, 0)
        model_population = gp_algorithm.get_population()
        evals = 5

        pop_options = [5, 10, 20, 50]
        generation_options = [5, 10, 20, 50]

        print 'pop, gen, mean, std, time'
        for p in pop_options:
            for g in generation_options:
                start = datetime.datetime.now()
                results = np.zeros((population_size_gp, evals))
                for i in range(0, evals):
                    print i
                    fitness_values_gp = self.evaluate_population_gp(model_population, cache=False, pop_size=p, generations=g)
                    results[:,i] = np.array(fitness_values_gp)
                end = datetime.datetime.now()
                print p, g, np.mean(np.mean(results, axis=1)), np.mean(np.std(results, axis=1)), (end-start).microseconds


    def run_gp(self, states, population_size_gp, generations, max_model_depth,
               training_data, test_data, max_params, eval_aspects, parallel=False, processors=1):
        self.training_data = training_data
        self.test_data = test_data
        self.eval_aspects = eval_aspects
        self.output_directory = self.util.create_output_directory()
        self.parallel = parallel
        print '-----'
        print mp.cpu_count()
        self.processors = (mp.cpu_count()-1)

        # Create the initial population of models in the GP

        gp_algorithm = GP()
        gp_algorithm.initialize(states, population_size_gp, max_model_depth, max_params, 0)

        for i in range(0, generations):
            print 'generation number: ' + str(i)
            model_population = gp_algorithm.get_population()
            fitness_values_gp = self.evaluate_population_gp(model_population)
            gp_algorithm.set_fitness(fitness_values_gp)
            best_individual_gp = gp_algorithm.evolve_population()
            self.util.write_results_to_file(self.output_directory, fitness_values_gp, best_individual_gp, i)

        return [best_individual_gp, self.output_directory]