import numpy as np
import scipy.linalg
from scipy.stats import pearsonr
from sklearn import linear_model, datasets


class Regression():

    def find_correlations(self, inputs, input_attributes, output):
        pearsons = []
        for input in range(inputs.shape[1]):
            if sum(np.asarray(inputs[:,input])) != 0:
                p = pearsonr(np.asarray(inputs[:,input]).tolist(), np.asarray(output).tolist())
                pearsons.append(p[0])
            else:
                pearsons.append(0)
        return pearsons


    def train_regression(self, training_inputs, input_attributes, training_outputs, output_attributes):
        if len(output_attributes) > 1:
            'Ignoring outputs other than [0] for now'
        new_training_outputs = []
        for t_o in range(len(training_outputs)):
            new_training_outputs.append(training_outputs[t_o][0])
        new_output_attributes = output_attributes[0]

        pearsons = []
        inputs = np.array(training_inputs)
        pearsons = self.find_correlations(inputs, input_attributes, new_training_outputs)

        outputs = np.array(new_training_outputs)
        self.model = linear_model.Ridge(alpha = 0.5)
        self.model.fit(inputs, outputs)

    def test_regression(self, test_inputs, test_outputs):
        return self.model.predict(test_inputs).tolist()