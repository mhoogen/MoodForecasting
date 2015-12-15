import csv
from dateutil import parser
import time
import numpy

class Util():

    default_nan = numpy.nan

    def read_file(self, file, delim):
        dataset = {}
        reader = csv.reader(open(file, 'rb'), delimiter=delim, quotechar='"')
        # Skip the header
        reader.next()
        for row in reader:
            # Assume a structure of measurement, id, time, variable, value
            id = row[1]
            t = parser.parse(row[2])
            variable = row[3]
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
