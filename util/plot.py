import matplotlib.pyplot as plt
import numpy

class Plot():


    def select_values(self, xcoord, ycoord):
        indices = [index for index,value in enumerate(ycoord) if not numpy.isnan(value)]
        new_xcoord = []
        new_ycoord = []
        for i in indices:
            new_xcoord.append(xcoord[i])
            new_ycoord.append(ycoord[i])
        return new_xcoord, new_ycoord

    def plot_initial_dataset(self, dataset, id):
        #First get the time points from the dataset, these are the x-coordinates

        xcoord = dataset[id]['times']
        # Scale time to hours

        new_xcoord = []
        leg = []

        for x in xcoord:
            new_xcoord.append((x - min(xcoord))/float(60*60))

        for var in dataset[id]:
            if not var == 'times':
                leg.append(var)
                ycoord = dataset[id][var]
                x, y = self.select_values(new_xcoord, ycoord)
                plt.plot(x, y, '-')

        plt.legend(leg)
        plt.ylabel('Value')
        plt.xlabel('Time (hours)')
        plt.show()

    def plot_results(self, desired_output, pred_output, name_output, plot_name):
        if not (type(pred_output[0]) == list):
            for i in range(len(pred_output)):
                pred_output[i] = [pred_output[i]]
        plt.figure(plot_name)
        num_outputs = len(desired_output[0])
        x = range(0,len(desired_output))
        leg  = []

        for i in range(num_outputs):
            desired_y = []
            pred_y = []
            leg.append('desired_' + name_output[i])
            leg.append('predicted_' + name_output[i])

            for j in range(len(desired_output)):
                desired_y.append(desired_output[j][i])
                pred_y.append(pred_output[j][i])
            plt.plot(x, desired_y, '-')
            plt.plot(x, pred_y, ':')

        plt.legend(leg)
        plt.ylabel('Value')
        plt.xlabel('Timesteps')
        plt.show()