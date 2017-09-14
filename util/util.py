import csv
from dateutil import parser
import time
import numpy
#from sklearn.metrics import mean_squared_error
from math import sqrt
import model.model
import os
import matplotlib.pyplot as plt
from eval.formal_method import EvaluationFramework
import datetime
from operator import itemgetter
import numpy as np

class Util():

    default_nan = numpy.nan

    def create_output_directory(self):
        directory = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
        newpath = '../data/output_runs/' + directory + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        return newpath

    def attribute_number(self, description, attributes):
        index = 0
        for attr in attributes:
            if attr in description:
                return index
            else:
                index +=1
        return -1

    def read_file_ema_e_compared(self, file, delim):
        attributes = ['sleep', 'mood', 'worrying', 'selfesteem', 'enjoyed', 'social', 'pleasantactivitylevel']
        time_str = 'Time'
        id = 'Username'
        dataset = {}

        reader = csv.reader(open(file, 'rb'), delimiter=delim)
        header = reader.next()

        for row in reader:
            current_id = row[0]
            current_time = datetime.datetime.strptime((row[2] + ' ' + row[3]), '%Y-%m-%d %H:%M:%S')

            if not current_id in dataset:
                dataset[current_id] = {'times':[]}

                for attr in attributes:
                    dataset[current_id]['self.'+attr] = []

            if not current_time in dataset[current_id]['times']:
                dataset[current_id]['times'].append(current_time)
                for attr in attributes:
                    dataset[current_id]['self.'+attr].append(self.default_nan)
            dataset[current_id]['self.'+ attributes[int(row[5])-1]][dataset[current_id]['times'].index(current_time)] = float(row[6])

        # Now we need to order the time points for each of our datasets....
        sorted_dataset = {}
        for id in dataset:
            sorted_dataset[id] = {}
            sorted_index = sorted(range(len(dataset[id]['times'])),key=lambda x:dataset[id]['times'][x])
            sorted_dataset[id]['times'] = list(np.array(dataset[id]['times'])[sorted_index])
            for attr in attributes:
                sorted_dataset[id]['self.' + attr] = list(np.array(dataset[id]['self.' + attr])[sorted_index])
        return sorted_dataset

    def read_file_jeroen(self, file, delim):
        dataset = {}
        reader = csv.reader(open(file, 'rb'), delimiter=delim, quotechar='"')
        # Skip the header
        reader.next()
        for row in reader:
            # Assume a structure of measurement, id, time, variable, value
            id = row[1]
            t = parser.parse(row[2])
            variable = 'self.' + row[3].replace('.','')
            try:
                value = float(row[4])
            except:
                value = self.default_nan

            # If we have already seen the ID.
            if not id in dataset:
                dataset[id] = {'times': []}

            if not variable in dataset[id]:
                dataset[id][variable] = [self.default_nan] * len(dataset[id]['times'])

            # Check if the time is in there
            current_times = dataset[id]['times']
            if t in current_times:
                index = current_times.index(t)
                dataset[id][variable][index] = value
            else:
                # insert it into the right position
                index = self.index_time(t, current_times)
                for var in dataset[id]:
                    if var == 'times':
                        dataset[id][var].insert(index, t)
                    elif var == variable:
                        dataset[id][var].insert(index, value)
                    else:
                        dataset[id][var].insert(index, self.default_nan)
        return dataset

    def index_time(self, t, time_array):
        if (len(time_array) == 0):
            return 0

        current_time = t
        current_index = 0
        while current_index < len(time_array) and (time_array[current_index] < t):
            current_index += 1
        return current_index

    def mse(self, predicted, desired):
        mse = []
        for i in range(len(predicted[0])):
            pred = []
            des = []
            for j in range(len(predicted)):
                pred.append(predicted[j][i])
                des.append(desired[j][i])
            mse.append(np.mean((np.array(pred)-np.array(des))**2))

        return mse

    def write_results_to_file(self, directory, fitness_gp, fitness_ga, best_model, generation):
        f_model = open(directory+'models', 'a')
        best_model.print_model_to_file(f_model, generation)
        f_ga = open(directory+'ga.csv', 'a')
        f_ga.write(str(generation) + ', ' + str(max(fitness_ga)) + ', ' + str(numpy.mean(fitness_ga)) + ', ' + str(numpy.std(fitness_ga)) + ', ' + '\n')


    def write_results_to_file(self, directory, fitness_gp, best_model, generation):
        f_model = open(directory+'models', 'a')
        best_model.print_model_to_file(f_model, generation)
        f_gp = open(directory+'gp.csv', 'a')
        f_gp.write(str(generation) + ', ' + str(max(fitness_gp)) + ', ' + str(numpy.mean(fitness_gp)) + '\n')

        # And print a graph of the best individual somehow.

