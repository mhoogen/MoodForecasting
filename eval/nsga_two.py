from inspyred.ec import emo
import random
#from sklearn.metrics import mean_squared_error
import math
import numpy as np

class PatientProblem():

    training_data = {}
    test_data = {}
    model = []
    eval_aspects = []
    ra = []

    def __init__(self):
        self.training_data = {}
        self.test_data = {}
        self.model = []
        self.eval_aspects = []
        self.ra = []


    def set_values(self, m, training, test, eval_aspects, ra=[-1,1]):
        self.training_data = training
        self.test_data = test
        self.model = m
        self.eval_aspects = eval_aspects
        self.ra = ra

    def generator(self, random, args):
        numb_parameters = len(self.model.parameter_names)
        return [random.uniform(self.ra[0], self.ra[1]) for _ in range(numb_parameters)]

    def evaluator_internal(self, candidates, training=True):
        number_of_steps = 3
        fitness = []
        for c in candidates:
            #print c
            #print self.model.print_model()
            self.model.reset()

            if (training):
                data = self.training_data
            else:
                data = self.test_data

            predictions = {}
            real = {}

            for eval in self.eval_aspects:
                predictions[eval] = []
                real[eval] = []

            for step in range(len(data[self.model.state_names[0]])-number_of_steps):
                self.model.reset()
                state_values = []
                for i in range(len(self.model.state_names)):
                    if self.model.state_names[i] in data.keys():
                        state_values.append(data[self.model.state_names[i]][step])
                    else:
                        state_values.append(0.5)
                self.model.set_state_values(state_values)
                self.model.set_parameter_values(c)
                self.model.execute_steps(number_of_steps)

                for eval in self.eval_aspects:
                    pred = self.model.get_values(eval)
                    predictions[eval].extend(pred)
                    real[eval].extend(data[eval][1+step:1+step+number_of_steps])

            evals = []
            for eval in self.eval_aspects:
                mse = np.mean((np.array(predictions[eval])-np.array(real[eval]))**2)
                evals.append(mse)
            fitness.append(emo.Pareto(evals))
        return fitness


    def evaluator(self, candidates, args):
        return self.evaluator_internal(candidates, True)

    def predict(self, candidates):
        return self.evaluator_internal(candidates, False)