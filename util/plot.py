import matplotlib.pyplot as plt
import numpy
from eval.formal_method import EvaluationFramework


class Plot():

    legends = {}

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

    def visualize_real_values(self, training_data, test_data, validation_data, eval_aspects, individual, fig_name, save=False, directory=''):

        self.legends = {}

        for eval in eval_aspects:
            ref_fig = plt.figure(fig_name + '_' + eval)
            plt.title('Performance of model for ' + eval.replace('self.', ''))
            real_values_training = training_data[individual][eval]
            real_values_test = test_data[individual][eval]
            real_values_validation = validation_data[individual][eval]
            time_training = list(range(0, len(real_values_training)))
            time_test = list(range(len(real_values_training), len(real_values_test)+len(real_values_training)))
            time_validation = list(range(len(real_values_training)+len(real_values_test), len(real_values_test)+len(real_values_training)+len(real_values_validation)))

            training_plot, = plt.plot(time_training, real_values_training, 'b--')
            test_plot, = plt.plot(time_test, real_values_test, 'r--')
            validation_plot, = plt.plot(time_validation, real_values_validation, 'k--')
            plt.ylabel('Value')
            plt.xlabel('Timesteps')
            #ax = ref_fig.add_subplot(111)
            #bbox_props = dict(boxstyle="doublearrow,pad=0.3", fc="cyan", ec="b", lw=2)
            plt.annotate('', xy=(time_training[0],1.1), xycoords='data', xytext=(time_training[-1], 1.1), textcoords='data', arrowprops={'arrowstyle': '<->'})
            plt.annotate('training set', xy=((float(sum(time_training))/len(time_training)), 1.05), color='blue', xycoords='data', ha='center')
            plt.annotate('', xy=(time_test[0],1.1), xycoords='data', xytext=(time_test[-1], 1.1), textcoords='data', arrowprops={'arrowstyle': '<->'})
            plt.annotate('validation set', xy=((float(sum(time_test))/len(time_test)), 1.05), color='red', xycoords='data', ha='center')
            plt.annotate('', xy=(time_validation[0],1.1), xycoords='data', xytext=(time_validation[-1], 1.1), textcoords='data', arrowprops={'arrowstyle': '<->'})
            plt.annotate('test set', xy=((float(sum(time_validation))/len(time_validation)), 1.05), xycoords='data', ha='center')
            plt.ylim(0,1.2)

            self.legends[eval] = {'data':[(training_plot, test_plot, validation_plot)], 'text':['measured values']}

            #t = ax.text(0, 1, "Training data", ha="center", va="center", rotation=0, size=(len(time_training)-1), bbox=bbox_props)
            #t = ax.text(len(time_training), 1, "Validation data", ha="center", va="center", rotation=0, size=(len(time_validation)-1), bbox=bbox_props)
            #t = ax.text((len(time_training)+len(time_validation)), 1, "Test data", ha="center", va="center", rotation=0, size=(len(time_test)-1), bbox=bbox_props)
            # plt.arrow(1, 1, 2, 1)
            if save:
                plt.savefig(directory + fig_name + '_' + eval.replace('self.', ''), bbox_inches='tight')
                plt.close(fig_name + '_' + eval)

    def visualize_performance(self, y_train_real, y_train_pred, y_test_real, y_test_pred, fig_name, directory):
            time_training = list(range(0, len(y_train_real)))
            time_test = list(range(len(y_train_real), len(y_train_real)+len(y_test_real)))

            plt.figure(fig_name)
            plt.hold(True)
            plt.plot(time_training, y_train_real, 'r--')
            plt.plot(time_test, y_test_real, 'b-')
            plt.plot(time_training, y_train_pred, 'ko', markersize=2)
            plt.plot(time_test, y_test_pred, 'ko', markersize=2)

            plt.ylabel('Value')
            plt.xlabel('Time')
            #ax = ref_fig.add_subplot(111)
            #bbox_props = dict(boxstyle="doublearrow,pad=0.3", fc="cyan", ec="b", lw=2)
            plt.annotate('', xy=(time_training[0],1.1), xycoords='data', xytext=(time_training[-1], 1.1), textcoords='data', arrowprops={'arrowstyle': '<->'})
            plt.annotate('training set', xy=((float(sum(time_training))/len(time_training)), 1.05), color='red', xycoords='data', ha='center')
            plt.annotate('', xy=(time_test[0],1.1), xycoords='data', xytext=(time_test[-1], 1.1), textcoords='data', arrowprops={'arrowstyle': '<->'})
            plt.annotate('test set', xy=((float(sum(time_test))/len(time_test)), 1.05), color='blue', xycoords='data', ha='center')
            plt.ylim(0,1.2)
            plt.legend(['training data', 'test data', 'predicted values'], loc=4, fontsize=7)

            plt.hold(False)
            # plt.show()

            # self.legends[eval_aspects[eval]]['data'].append((training_plot, test_plot, validation_plot))
            # self.legends[eval_aspects[eval]]['text'].append('echo state network')

            plt.savefig(directory + fig_name +'.png', bbox_inches='tight')
            plt.close(fig_name)


    def visualize_performance_gp(self, model, training_data, test_data, validation_data, eval_aspects, individual, fig_name, save=False, directory=''):


        ef = EvaluationFramework()
        best_params = ef.get_best_model_parameters(model, training_data, test_data, eval_aspects, individual)

        print 'visualizing performance of model for individual ' + str(individual)
        model.print_model()
        print 'with parameter settings '
        print best_params


        for eval in eval_aspects:

            plt.figure(fig_name + '_' + eval)
            model.reset()

            for step in range(0, len(training_data[individual][model.state_names[0]])-1):

                initial_state_values = []

                for i in range(len(model.state_names)):
                    initial_state_values.append(training_data[individual][model.state_names[i]][step])
                model.set_state_values(initial_state_values)
                model.set_parameter_values(best_params)
                model.execute_steps(1)

            pred_values_training = model.get_values(eval)
            real_values_training = training_data[individual][eval]

            model.reset()

            for step in range(0, len(test_data[individual][model.state_names[0]])-1):
                initial_state_values = []
                for i in range(len(model.state_names)):
                    initial_state_values.append(test_data[individual][model.state_names[i]][step])
                model.set_state_values(initial_state_values)
                model.set_parameter_values(best_params)
                model.execute_steps(1)

            pred_values_test = model.get_values(eval)
            real_values_test = test_data[individual][eval]

            model.reset()

            for step in range(0, len(validation_data[individual][model.state_names[0]])-1):
                initial_state_values = []
                for i in range(len(model.state_names)):
                    initial_state_values.append(validation_data[individual][model.state_names[i]][step])
                model.set_state_values(initial_state_values)
                model.set_parameter_values(best_params)
                model.execute_steps(1)

            pred_values_validation = model.get_values(eval)
            real_values_validation = validation_data[individual][eval]

            time_training = list(range(1, len(pred_values_training)+1))
            time_test = list(range(len(pred_values_training)+2, len(pred_values_test)+len(pred_values_training)+2))
            time_validation = list(range(len(pred_values_training)+len(pred_values_test)+3, len(pred_values_training)+len(pred_values_test)+len(pred_values_validation)+3))

            training_plot, = plt.plot(time_training, pred_values_training[0:len(pred_values_training)], 'gd')
            test_plot, = plt.plot(time_test, pred_values_test[0:len(pred_values_test)], 'gd')
            validation_plot, = plt.plot(time_validation, pred_values_validation[0:len(pred_values_test)], 'gd')
            self.legends[eval]['data'].append((training_plot, test_plot, validation_plot))
            self.legends[eval]['text'].append('genetic programming')
            if save:
                plt.savefig(directory + fig_name + '_' + eval.replace('self.', ''), bbox_inches='tight')
                plt.close(fig_name + '_' + eval)

    def visualize_performance_esn(self, esn, input_training, output_training, input_test, output_test, input_validation, output_validation, eval_aspects, fig_name, save=False, directory=''):

        pred_values_training_raw = esn.testNetwork(input_training, output_training)[1:]
        pred_values_test_raw = esn.testNetwork(input_test, output_test)[1:]
        pred_values_validation_raw = esn.testNetwork(input_validation, output_validation)[1:]

        for eval in range(0, len(eval_aspects)):
            plt.figure(fig_name + '_' + eval_aspects[eval])

            pred_values_training = []
            for i in range(0, len(pred_values_training_raw)):
                pred_values_training.append(pred_values_training_raw[i][eval])
            pred_values_test = []
            for i in range(0, len(pred_values_test_raw)):
                pred_values_test.append(pred_values_test_raw[i][eval])
            pred_values_validation = []
            for i in range(0, len(pred_values_validation_raw)):
                pred_values_validation.append(pred_values_validation_raw[i][eval])


            time_training = list(range(1, len(input_training)))
            time_test = list(range(len(input_training)+1, len(input_training)+len(input_test)))
            time_validation = list(range(len(input_training)+len(input_test)+1, len(input_training)+len(input_test)+len(input_validation)))

            #plt.plot(time_training, real_values_training, 'r--')
            training_plot, = plt.plot(time_training, pred_values_training, 'c*')

            #plt.plot(time_test, real_values_test, 'y-')
            test_plot, = plt.plot(time_test, pred_values_test, 'c*')

            validation_plot, = plt.plot(time_validation, pred_values_validation, 'c*')

            self.legends[eval_aspects[eval]]['data'].append((training_plot, test_plot, validation_plot))
            self.legends[eval_aspects[eval]]['text'].append('echo state network')


            leg = plt.legend(self.legends[eval_aspects[eval]]['data'], self.legends[eval_aspects[eval]]['text'], bbox_to_anchor=(1.2, 0.4), loc='upper center', fontsize = 'medium')
            leg.get_frame().set_linewidth(0.0)

            if save:
                plt.savefig(directory + fig_name + '_' + eval_aspects[eval].replace('self.', ''), bbox_inches='tight')
                plt.close(fig_name + '_' + eval_aspects[eval])
