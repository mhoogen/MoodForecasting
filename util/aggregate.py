import copy
import time
import datetime
import numpy as np

class Aggregate():

    default_nan = np.nan
    sum_attributes = ['appCat.office', 'appCat.communication', 'appCat.entertainment', 'appCat.utilities', 'appCat.builtin', 'appCat.weather', 'sms', 'appCat.game', 'appCat.travel', 'call', 'appCat.finance', 'appCat.other', 'activity', 'appCat.unknown', 'appCat.social', 'screen']

    def identify_regression_dataset(self, initial_dataset, id, training_frac, target, remove):
        [input_attributes, input_training, input_test, output_attributes, output_training, output_test] = self.identify_rnn_dataset(initial_dataset, id, training_frac, target, remove)
        for target_output in range(len(output_attributes)):
            input_attributes.append('prev_' + output_attributes[target_output])
            for i in range(len(input_training)):
                if i > 0:
                    input_training[i].append(output_training[i-1][target_output])
                else:
                    input_training[i].append(0)
            for i in range(len(input_test)):
                if i > 0:
                    input_test[i].append(output_test[i-1][target_output])
                else:
                    input_test[i].append(0)
        return input_attributes, input_training, input_test, output_attributes, output_training, output_test

    def identify_rnn_dataset(self, initial_dataset, id, training_frac, target, remove):

        # Training set smaller than the test set not allowed.
        if training_frac < 0.5:
            return

        input_training = [] # Input with for each element the input at that time point.
        output_training = []
        input = []
        output = []
        input_attributes = []
        output_attributes = []
        training_set_len = int(training_frac * len(initial_dataset[id]['times']))
        init = True

        for index in range(len(initial_dataset[id]['times'])):
            if index == training_set_len:
                input_training = input
                output_training = output
                input = []
                output = []
            input.append([])
            output.append([])
            for attribute in initial_dataset[id]:
                if attribute in target:
                    if init:
                        output_attributes.append(attribute)
                    output[index % training_set_len].append(initial_dataset[id][attribute][index])
                elif not attribute == 'times' and not attribute in remove:
                    if init:
                        input_attributes.append(attribute)
                    input[index % training_set_len].append(initial_dataset[id][attribute][index])
            init = False
        if training_set_len == len(initial_dataset[id]['times']):
            return input_attributes, input, [], output_attributes, output, []
        else:
            return input_attributes, input_training, input, output_attributes, output_training, output

    def impute_nan(self, initial_dataset):
        dataset = copy.deepcopy(initial_dataset)
        for id in dataset:
            for attribute in dataset[id]:
                np_dataset = np.array(dataset[id][attribute])
                mask = np.isnan(np_dataset)
                np_dataset[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), np_dataset[~mask])
                dataset[id][attribute] = np_dataset.tolist()
        return dataset

    def normalize_data(self, initial_dataset):
        dataset = copy.deepcopy(initial_dataset)
        for id in initial_dataset:
            for attribute in dataset[id]:
                if not attribute == 'times':
                    range = ((max(dataset[id][attribute]))-min(dataset[id][attribute]))
                    if range == 0:
                        range = 1
                    dataset[id][attribute] = [(float(i)-min(dataset[id][attribute]))/range for i in dataset[id][attribute] ]
        return dataset

    # Take the dataset and aggregate with a certain granularity (in minutes)
    def aggregate_dataset(self, dataset, granularity):
        new_dataset = {}
        for id in dataset:

            new_dataset[id] = {}
            current_aggr_set = {}

            for var in dataset[id]:
                new_dataset[id][var] = []
                current_aggr_set[var] = []

            # to synchronize, we start the day at 0:00.
            tp = copy.deepcopy(dataset[id]['times'][0])
            tp = tp.replace(hour = 0, minute = 0, second = 0)

            for i in range(0, len(dataset[id]['times'])+1):
                if ((i == len(dataset[id]['times'])) or
                    (time.mktime(dataset[id]['times'][i].timetuple()) > time.mktime(tp.timetuple()) + (granularity*60))):
                    # Time to aggregate!
                    aggregate_values = self.aggregate_period(tp, current_aggr_set, type)
                    for var in aggregate_values:
                        new_dataset[id][var].append(aggregate_values[var])
                        current_aggr_set[var] = []
                    tp = datetime.datetime.fromtimestamp(time.mktime(tp.timetuple()) + granularity*60)

                    if (i == len(dataset[id]['times'])):
                        break
                    while not time.mktime(dataset[id]['times'][i].timetuple()) < time.mktime(tp.timetuple()) + (granularity*60):
                        aggregate_values = self.aggregate_period(tp, current_aggr_set, type)
                        for var in aggregate_values:
                            new_dataset[id][var].append(aggregate_values[var])
                        tp = datetime.datetime.fromtimestamp(time.mktime(tp.timetuple()) + granularity*60)
                for var in current_aggr_set:
                    current_aggr_set[var].append(dataset[id][var][i])
        return new_dataset

    def aggregate_period(self, timepoint, values, type):
        data = True
        aggregate_values = {}
        aggregate_values['times'] = time.mktime(timepoint.timetuple())
        if len(values['times']) == 0:
            # We don't have any data....
            data = False

        for var in values:
            if not var == 'times':
                if data:
                    selected_values = [value for index,value in enumerate(values[var]) if not np.isnan(value)]
                    if len(selected_values) > 0:
                        if var in self.sum_attributes:
                            aggregate_values[var] = sum(selected_values)
                        else:
                            aggregate_values[var] = sum(selected_values)/float(len(selected_values))
                    else:
                        aggregate_values[var] = self.default_nan
                else:
                    aggregate_values[var] = self.default_nan
        return aggregate_values