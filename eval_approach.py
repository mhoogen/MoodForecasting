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

def generate_lstm_prediction(training_data, validation_data, individual, eval_aspects):

    b_size = 7
    x_train, y_train = ag.identify_lstm_dataset(training_data, individual, eval_aspects, prediction_time)
    x_test, y_test = ag.identify_lstm_dataset(validation_data, individual, eval_aspects, prediction_time)
    model = Sequential()
    # model.add(Embedding(len(training_data[individual].values()),
    #                    output_dim=(len(eval_aspects)*prediction_time)))
    # model.add(Embedding(1000, 64, input_length=10))
    model.add(LSTM(units=128, activation = 'sigmoid', input_shape = (1, x_train.shape[1])))
    model.add(Dense((len(eval_aspects)*prediction_time), activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=[metrics.mse])
    model.fit(x_train.reshape(x_train.shape[0], 1, x_train.shape[1]), y_train, batch_size=b_size, epochs=5)
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
                                     'lstm_'+individual+'_'+eval+'(t+'+str((j+1))+')', output_directory)
        rmse.append(current_rmse)
        i = i + 1
    print rmse
    return rmse

def predict_using_model(model, best_params, data, individual, eval_aspects):
    pred_values = np.zeros((len(data[individual][model.state_names[0]])-prediction_time-1, prediction_time * len(eval_aspects)))
    real_values = np.zeros((len(data[individual][model.state_names[0]])-prediction_time-1, prediction_time * len(eval_aspects)))
    for step in range(0, len(data[individual][model.state_names[0]])-prediction_time-1):
        initial_state_values = []

        for i in range(len(model.state_names)):
            initial_state_values.append(data[individual][model.state_names[i]][step])

        model.reset()
        model.set_state_values(initial_state_values)
        model.set_parameter_values(best_params)
        model.execute_steps(prediction_time)

        for i in range(0, len(eval_aspects)):
            pred_values[step, i*prediction_time:(i+1)*prediction_time] = model.get_values(eval_aspects[i])
            real_values[step, i*prediction_time:(i+1)*prediction_time] = data[individual][eval_aspects[i]][step+1:step+1+prediction_time]
    return pred_values, real_values


def generate_model_prediction(model, training_data, training_full, test_data, validation_data, individual, eval_aspects):
    ef = EvaluationFramework()
    best_params = ef.get_best_model_parameters(model, training_data, test_data, eval_aspects, individual)

    y_train_pred, y_train_real = predict_using_model(model, best_params, training_full, individual, eval_aspects)
    y_test_pred, y_test_real = predict_using_model(model, best_params, validation_data, individual, eval_aspects)

    i = 0
    rmse = []
    for eval in eval_aspects:
        current_rmse = []
        for j in range(0, prediction_time):
            current_rmse.append(math.sqrt(sklearn.metrics.mean_squared_error(y_test_real[:,i*prediction_time+j],
                                                            y_test_pred[:,i*prediction_time+j])))
            pl.visualize_performance(y_train_real[:,i*prediction_time+j],
                                     y_train_pred[:,i*prediction_time+j],
                                     y_test_real[:,i*prediction_time+j],
                                     y_test_pred[:,i*prediction_time+j],
                                     'gp_'+individual+'_'+eval+'(t+'+str((j+1))+')', output_directory)
        rmse.append(current_rmse)
        i = i + 1
    print rmse
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
result_selected = ag.select_longest_period(result, max_missing_values, 'self.mood')
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

[training_is_gp, test_is_gp, validation_is_gp, states] = ag.identify_gp_dataset(results_in_sample, training_frac, test_frac, validation_frac, [])
[training_os_gp, test_os_gp, validation_os_gp, states] = ag.identify_gp_dataset(results_out_sample, training_frac, test_frac, validation_frac, [])
[training_is_gp_full, test_is_gp_full, validation_is_gp_full, states_full] = ag.identify_gp_dataset(results_in_sample, training_frac + test_frac, 0, validation_frac, [])
[training_os_gp_full, test_os_gp_full, validation_os_gp_full, states_full] = ag.identify_gp_dataset(results_out_sample, training_frac + test_frac, 0, validation_frac, [])
[training_is_lstm, test_is_lstm, validation_is_lstm, states] = ag.identify_gp_dataset(results_in_sample, training_frac + test_frac, 0, validation_frac, [])
[training_os_lstm, test_os_lstm, validation_os_lstm, states] = ag.identify_gp_dataset(results_out_sample, training_frac + test_frac, 0, validation_frac, [])

print 'done.'


# Define the three models:

cols = []
algs = ['gp', 'lstm']

for alg in algs:
    for eval in eval_aspects:
        for t in range(0, prediction_time):
            cols.append(alg + '_' + eval + '_(t+' + str(t+1) + ')')


results_in_sample = pd.DataFrame(0, index=training_is_gp.keys(), columns=cols)

i = 0
for individual in training_is_gp.keys():
#for individual in ['0102055']:

    # 1: Literature model



    # 2: GP model

    print 'GP....'

    gp_model = Model()
    gp_model.set_model(['self.worrying', 'self.mood', 'self.social', 'self.sleep', 'self.pleasantactivitylevel',
                        'self.enjoyed', 'self.selfesteem'],
                        ['((((self.enjoyed-self.selfesteem)*((self.enjoyed-self.selfesteem)*(self.social-self.worrying)))*(self.social-self.worrying))*(self.social-self.worrying))',
                         'self.enjoyed', 'self.sleep', 'self.sleep','self.enjoyed',
                         '((self.selfesteem*(self.param1v+self.pleasantactivitylevel))-self.param1v)','self.social'],
                         ['self.param1v'])
    error_gp = generate_model_prediction(gp_model, training_is_gp, training_is_gp_full, test_is_gp, validation_is_gp, individual, eval_aspects)
    for j in range(0, len(eval_aspects)):
        results_in_sample.ix[i, j*prediction_time:(j+1)*prediction_time] = error_gp[j]
#    print error_gp

    # 3: LSTM

    print 'LSTM....'
    error_lstm = generate_lstm_prediction(training_is_lstm, validation_is_lstm, individual, eval_aspects)
    for j in range(0, len(eval_aspects)):
        results_in_sample.ix[i, (len(eval_aspects)*prediction_time) +
                             j*prediction_time:(len(eval_aspects)*prediction_time) + (j+1)*prediction_time] = error_lstm[j]    print error_lstm
    i = i + 1

results_in_sample.to_csv(output_directory + 'results_in_sample.csv')
