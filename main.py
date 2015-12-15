# Read the data from the

import csv
from util.util import Util
from util.aggregate import Aggregate
from util.plot import Plot
from learning.echo_state_network import EchoStateNetwork
from learning.regression import Regression

from dateutil import parser

granularity = 60*6
u = Util()
print 'reading the file....',
dataset = u.read_file('../data/LKW2015_patient_1.csv', ',')
#dataset = u.read_file('../data/test.csv', ',')
print 'done.'
ag = Aggregate()

print 'aggregating the data'
print 'aggregating time...',
result = ag.aggregate_dataset(dataset, granularity)
print 'done.'
print 'imputing nan and normalizing set...',
result_nan = ag.impute_nan(result)
result_norm = ag.normalize_data(result_nan)
print 'done.'
print 'constructing training and test set...',
#input_attributes, input_training, input_test, output_attributes, output_training, output_test = ag.identify_rnn_dataset(result_norm, 'AS14.01', 0.5, ['mood'], ['circumplex.arousal', 'circumplex.valence'])
input_attributes, input_training, input_test, output_attributes, output_training, output_test = ag.identify_regression_dataset(result_norm, 'AS14.01', 0.5, ['mood'], ['circumplex.arousal', 'circumplex.valence'])
print ' done.'

print 'regression...',
regr = Regression()
regr.train_regression(input_training, input_attributes, output_training, output_attributes)
Y_regression = regr.test_regression(input_test, output_test)
print 'done'

print 'echo state network'
esn = EchoStateNetwork()
esn.initializeNetwork(len(input_attributes), len(output_attributes), 10, True)
print 'training the network...',
esn.trainNetwork(input_training, output_training)
print 'done.'
print 'testing the network...',
Y_esn = esn.testNetwork(input_test, output_test)
print 'done.'

p1 = Plot()
p1.plot_results(output_test, Y_regression, output_attributes, 'regression')
p2 = Plot()
p2.plot_results(output_test, Y_esn, output_attributes, 'ESN')