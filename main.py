# Read the data from the

import csv
from util.util import Util
from util.aggregate_dataset import Aggregate
from util.plot import Plot
from learning.echo_state_network import EchoStateNetwork
from learning.regression import Regression
from learning.benchmark import Benchmark
from model.model import Model
from eval.formal_method import EvaluationFramework
from learning.ea import EA
import datetime


from dateutil import parser

granularity = 60*24
min_days_per_patient = 40
number_patients = 10
max_missing_values = 2
u = Util()
print 'reading the file....',
dataset = u.read_file_ema_e_compared('../data/ema_data.csv', ',')
print 'done.'
ag = Aggregate()

print 'aggregating the data'
print 'aggregating time...',
result = ag.aggregate_dataset(dataset, granularity)
result_selected = ag.select_longest_period(result, max_missing_values, 'self.mood')
print 'done.'
print 'imputing nan and normalizing set...'
result_nan = ag.impute_nan(result)
result_norm = ag.normalize_data(result_nan)
result_selected = ag.filter_datalack_cases(result_norm, min_days_per_patient)
#result_limited = ag.select_max_patients(result_selected, number_patients)
print 'done.'
print 'constructing training and test set...',
training_frac = 0.6
test_frac = 0.2
validation_frac = 1 - training_frac - test_frac

[training, test, validation, states] = ag.identify_gp_dataset(result_selected, training_frac, test_frac, validation_frac, [])
#input_attributes, input_training, input_test, output_attributes, output_training, output_test = ag.identify_regression_dataset(result_norm, 'AS14.01', 0.5, ['mood'], ['circumplex.arousal', 'circumplex.valence'])
print ' done.'

population_size_gp = 100
#population_size_ga = 1
generations = 100
max_depth = 6
max_params = 7
eval_aspects = ['self.mood', 'self.sleep']

ea = EA()
print 'Testing the parameters for the evaluation framework....'
##ea.run_coev(states, population_size_gp, population_size_ga, generations, max_depth, training, test, max_params, ['self.mood'], 0.1, 0.5, True, 4)
# ea.evaluate_params_nsga_2(states, population_size_gp, generations, max_depth, training, test, max_params, eval_aspects, True, 3)

print datetime.datetime.now()

[best_individual, output_directory] = ea.run_gp(states, population_size_gp, generations, max_depth, training, test, max_params, eval_aspects, True, 3)

print datetime.datetime.now()


for individual in training:
    figure_name = individual
    p = Plot()
    p.visualize_real_values(training, test, validation, eval_aspects, individual, figure_name)
    [input_attributes, input_training, input_test, input_validation, output_attributes, output_training, output_test, output_validation] = ag.identify_rnn_dataset(result_norm, individual, training_frac, test_frac, validation_frac, eval_aspects, [])
    p.visualize_performance_gp(best_individual, training, test, validation, eval_aspects, individual, figure_name)

#
#    ## Echo state model
#    print 'echo state network for ' + str(individual)
#    esn = EchoStateNetwork()
#    esn.initializeNetwork(len(input_attributes), len(output_attributes), 10, True)
#    print 'training the network...',
#    esn.trainNetwork(input_training, output_training)
#    print 'done.'
#
#    p.visualize_performance_esn(esn, input_training, output_training, input_test, output_test, input_validation, output_validation, eval_aspects, individual, output_directory)
#    print 'done.'
#print 'done.'

# Regression model
#print 'regression...',
#regr = Regression()
#regr.train_regression(input_training, input_attributes, output_training, output_attributes)
#Y_regression = regr.test_regression(input_test, output_test)
#print 'done'
#
#
## Benchmark model (feed last mood forward)
#print 'benchmark...',
#bm = Benchmark()
#Y_bench = bm.test_benchmark(output_test)
#print 'done.'
#
#print 'performance benchmark: ',
#print u.mse(Y_bench, output_test)
#print 'performance regression: ',
#print u.mse(Y_regression, output_test)
#print 'performance esn: ',
#print u.mse(Y_esn, output_test)
#
#p1 = Plot()
#p1.plot_results(output_test, Y_regression, output_attributes, 'Regression')
#p2 = Plot()
#p2.plot_results(output_test, Y_esn, output_attributes, 'ESN')
#p3 = Plot()
#p3.plot_results(output_test, Y_bench, output_attributes, 'Benchmark')