# Read the data from the

import csv
import math
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
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import metrics
import numpy as np
import pandas as pd

ag = Aggregate()
prediction_time = 3
pl = Plot()
util = Util()
output_directory = util.create_output_directory()

def create_lstm_prediction(training_data, eval_aspects):

    b_size = 7

    x_train = []
    y_train = []
    for individual in training_data.keys():
        if x_train == []:
            x_train, y_train = ag.identify_lstm_dataset(training_data, individual, eval_aspects, prediction_time)
        else:
            x_train_extra, y_train_extra = ag.identify_lstm_dataset(training_data, individual, eval_aspects, prediction_time)
            np.append(x_train, x_train_extra, axis=0)
            np.append(y_train, y_train_extra, axis=0)

    model = Sequential()
    model.add(LSTM(units=128, activation = 'sigmoid', input_shape = (1, x_train.shape[1])))
    model.add(Dense((len(eval_aspects)*prediction_time), activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=[metrics.mse])
    model.fit(x_train.reshape(x_train.shape[0], 1, x_train.shape[1]), y_train, batch_size=b_size, epochs=30)
    return model

def lstm_predict_using_model(model, training_data, validation_data, individual, eval_aspects):
    b_size = 7
    x_train, y_train = ag.identify_lstm_dataset(training_data, individual, eval_aspects, prediction_time)
    x_test, y_test = ag.identify_lstm_dataset(validation_data, individual, eval_aspects, prediction_time)

    predictions_train = model.predict(x_train.reshape(x_train.shape[0], 1, x_train.shape[1]), batch_size=b_size)
    predictions = model.predict(x_test.reshape(x_test.shape[0], 1, x_test.shape[1]), batch_size=b_size)

    i = 0
    rmse = []
    for eval in eval_aspects:
        current_rmse = []
        for j in range(0, prediction_time):
            current_rmse.append(math.sqrt(sklearn.metrics.mean_squared_error(y_test[:,i*prediction_time+j],
                                                            predictions[:,i*prediction_time+j])))
            pl.visualize_performance(y_train[:,i*prediction_time+j],
                                     predictions_train[:,i*prediction_time+j],
                                     y_test[:,i*prediction_time+j],
                                     predictions[:,i*prediction_time+j],
                                     'lstm_generic_'+individual+'_'+eval+'(t+'+str((j+1))+')', output_directory)
        rmse.append(current_rmse)
        i = i + 1
    return rmse



granularity = 60*24
min_days_per_patient = 40
in_sample_patients = ['0102055', '0800004', '0102448', '0200084', '0200143', '0800023', '0200077',
                     '0604012', '0200036', '0102423', '0200058', '0302018', '0800031', '0606006',
                     '0200012', '0102459', '0301051', '0102219', '0604006', '0800055', '0500079',
                     '0200141', '0604008', '0200096', '0102259', '0102162', '0702001', '0703023',
                     '0102366', '0800033']
max_missing_values = 2
u = Util()

print 'Evaluation....'
print 'reading the file....',
dataset = u.read_file_ema_e_compared('../data/ema_data.csv', ',')
print 'done.'

print 'aggregating the data'
print 'aggregating time...',
result = ag.aggregate_dataset(dataset, granularity)
print 'done.'
print 'imputing nan and normalizing set...'
result_nan = ag.impute_nan(result)
result_norm = ag.normalize_data(result_nan)
result_selected = ag.filter_datalack_cases(result_norm, min_days_per_patient)

results_in_sample = dict((key,value) for key, value in result_selected.iteritems() if key in in_sample_patients)
remaining_patients = dict((key,value) for key, value in result_selected.iteritems() if not (key in in_sample_patients))

number_patients = 30
results_out_sample = ag.select_max_patients(remaining_patients, number_patients)
#result_limited = result_selected

print 'done.'
print 'constructing training and test set...',
training_frac = 0.6
test_frac = 0.2
validation_frac = 1 - training_frac - test_frac
eval_aspects = ['self.mood', 'self.sleep']

[training_is_lstm, test_is_lstm, validation_is_lstm, states] = ag.identify_gp_dataset(results_in_sample, training_frac + test_frac, 0, validation_frac, [])
[training_os_lstm, test_os_lstm, validation_os_lstm, states] = ag.identify_gp_dataset(results_out_sample, training_frac + test_frac, 0, validation_frac, [])

print 'done.'


# Define the three models:

cols = []
algs = ['lit_generic']

for alg in algs:
    for eval in eval_aspects:
        for t in range(0, prediction_time):
            cols.append(alg + '_' + eval + '_(t+' + str(t+1) + ')')


results_in_sample = pd.DataFrame(0, index=training_is_lstm.keys(), columns=cols)

print 'in sample'
print 'sample is ', training_is_lstm.keys()

i = 0

model = create_lstm_prediction(training_is_lstm, eval_aspects)

for individual in training_is_lstm.keys():

    print 'Individual ', individual

    # 4: LSTM generic

    print 'LSTM. Generic...'
    error_lstm = lstm_predict_using_model(model, training_is_lstm, validation_is_lstm, individual, eval_aspects)
    for j in range(0, len(eval_aspects)):
        results_in_sample.ix[i, j*prediction_time:(j+1)*prediction_time] = error_lstm[j]

    print error_lstm
    i = i + 1

results_in_sample.to_csv(output_directory + 'results_in_sample.csv')

results_out_sample = pd.DataFrame(0, index=training_os_lstm.keys(), columns=cols)

print 'out of sample'
print 'sample is ', training_os_lstm.keys()

i = 0
for individual in training_os_lstm.keys():

    print 'Individual ', individual

    # 4: LSTM generic

    print 'LSTM. Generic...'
    error_lstm = lstm_predict_using_model(model, training_os_lstm, validation_os_lstm, individual, eval_aspects)
    for j in range(0, len(eval_aspects)):
        results_out_sample.ix[i, j*prediction_time:(j+1)*prediction_time] = error_lstm[j]

    print error_lstm
    i = i + 1

results_out_sample.to_csv(output_directory + 'results_out_of_sample.csv')
