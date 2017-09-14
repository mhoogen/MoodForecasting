import copy
import time
import datetime
import numpy as np
import random

class Aggregate():

    default_nan = np.nan

    sum_attributes = ['appCatoffice', 'appCatcommunication', 'appCatentertainment', 'appCatutilities', 'appCatbuiltin', 'appCatweather', 'sms', 'appCatgame', 'appCattravel', 'call', 'appCatfinance', 'appCatother', 'activity', 'appCatunknown', 'appCatsocial', 'screen']

    def identify_gp_dataset(self, initial_dataset, training_frac, test_frac, validation_frac, remove):
        trainingsset = copy.deepcopy(initial_dataset)
        testset = copy.deepcopy(initial_dataset)
        validationset = copy.deepcopy(initial_dataset)
        states = []
        init = True
        for id in initial_dataset:
            training_set_len = int(training_frac * len(initial_dataset[id]['times']))
            test_set_len = int(test_frac * len(initial_dataset[id]['times']))
            validation_set_len = min(int(validation_frac * len(initial_dataset[id]['times'])), (len(initial_dataset[id]['times'])-training_set_len-test_set_len))
            del trainingsset[id]['times']
            del testset[id]['times']
            del validationset[id]['times']
            for attr in trainingsset[id]:
                trainingsset[id][attr] = trainingsset[id][attr][0:training_set_len]
                testset[id][attr] = testset[id][attr][training_set_len:training_set_len+test_set_len]
                validationset[id][attr] = validationset[id][attr][training_set_len + test_set_len:training_set_len + test_set_len + validation_set_len]
                if init:
                    states.append(attr)
            init = False
        return [trainingsset, testset, validationset, states]


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

    def identify_rnn_dataset(self, initial_dataset, id, training_frac, test_frac, validation_frac, target, remove):

        # Training set smaller than the test set not allowed.
        if training_frac < 0.5:
            return

        input_training = [] # Input with for each element the input at that time point.
        output_training = []
        input_test = []
        output_test = []
        input = []
        output = []
        input_attributes = []
        output_attributes = []
        training_set_len = int(training_frac * len(initial_dataset[id]['times']))
        test_set_len = int(test_frac * len(initial_dataset[id]['times']))
        validation_set_len = min(int(validation_frac * len(initial_dataset[id]['times'])), (len(initial_dataset[id]['times'])-training_set_len-test_set_len))

        init = True

        index_in_set = 0
        for index in range(len(initial_dataset[id]['times'])-1):
            if index == training_set_len:
                input_training = input
                output_training = output
                input = []
                output = []
                index_in_set = 0
            if index == training_set_len + test_set_len:
                input_test = input
                output_test = output
                input = []
                output = []
                index_in_set = 0
            input.append([])
            output.append([])
            for attribute in initial_dataset[id]:
                if attribute in target:
                    if init:
                        output_attributes.append(attribute)
                    output[index_in_set].append(initial_dataset[id][attribute][index])
                elif not attribute == 'times' and not attribute in remove:
                    if init:
                        input_attributes.append(attribute)
                    input[index_in_set].append(initial_dataset[id][attribute][index])
            init = False
            index_in_set += 1

        if training_set_len == len(initial_dataset[id]['times']):
            return input_attributes, input, [], output_attributes, output, []
        else:
            return input_attributes, input_training, input_test, input, output_attributes, output_training, output_test, output

    def rle(self, arr, max_missing_values):
        #stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        arr = np.array(arr)
        assert len(arr.shape) == 1
        assert arr.dtype == np.bool
        if arr.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        sw = np.insert(arr[1:] ^ arr[:-1], [0, arr.shape[0]-1], values=True)
        swi = np.arange(sw.shape[0])[sw]
        offset = 0 if arr[0] else 1
        lengths = swi[offset+1::2] - swi[offset:-1:2]
        start_end = []

        i = offset
        while i < (len(swi)-1):
            if lengths[(i-offset)/2] > max_missing_values:
                start_end.append((swi[i], swi[i+1]))
            i = i + 2
        return start_end

    def select_longest_period(self, initial_dataset, max_missing_values, attribute):
        new_dataset = {}
        for id in initial_dataset:
            new_dataset[id] = {}
            nan_list = list(np.isnan(np.array(initial_dataset[id][attribute])))
            current_start_index = 0
            current_end_index = 0
            nan_sequence = 0
            lengths = []
            start_end = []

            sequences = self.rle(nan_list, max_missing_values)

            start_index = 0
            end_index = len(initial_dataset[id][attribute])
            indices = []
            lengths = []

            if len(sequences) > 0:
                for i in range(len(sequences)):
                    end_index = list(sequences[i])[0]
                    length = end_index - start_index
                    indices.append((start_index, end_index))
                    lengths.append(length)
                    start_index = list(sequences[i])[1]
                    end_index = start_index
            else:
                indices.append((start_index, end_index))
                length = end_index - start_index
                lengths.append(length)

            best_index = lengths.index(max(lengths))
            best_start = list(indices[best_index])[0]
            best_end = list(indices[best_index])[1]

            for attr in new_dataset[id]:
                new_dataset[id][attr] = list(np.array(initial_dataset[id][attr])[range(best_start, best_end)])
        return new_dataset

    def impute_nan(self, initial_dataset):
        dataset = copy.deepcopy(initial_dataset)
        delete_list = []
        for id in dataset:
            for attribute in dataset[id]:
                np_dataset = np.array(dataset[id][attribute])
                if (np.count_nonzero(~np.isnan(np_dataset)) < 2) or (len(np_dataset) <= 1 and np.isnan(np_dataset[0])):
                    delete_list.append(id)
                    break
                mask = np.isnan(np_dataset)
                np_dataset[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), np_dataset[~mask])
                dataset[id][attribute] = np_dataset.tolist()

        for id in delete_list:
            del dataset[id]
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

    def filter_datalack_cases(self, initial_dataset, threshold):
        new_dataset = copy.deepcopy(initial_dataset)
        for id in initial_dataset:
            if len(initial_dataset[id]['times']) < threshold:
                del new_dataset[id]
                print str(id) + ' removed'
        print 'initial number of patients: ' + str(len(initial_dataset))
        print 'selected number of patients: ' + str(len(new_dataset))
        return new_dataset


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

    def select_max_patients(self, dataset, max):
        new_dataset = {}
        random.seed(1)
        selection = random.sample(dataset.keys(), max)

        for id in selection:
            new_dataset[id] = dataset[id]
        return new_dataset