import math

class Model():

    state_names = []
    state_values = []
    predicted_values = []
    state_equations = []
    parameter_names = []
    parameter_values = []
    t = 0
    max_value = 10000

    def __init__(self):
        self.state_names = []
        self.state_values = []
        self.predicted_values = []
        self.state_equations = []
        self.parameter_names = []
        self.parameter_values = []
        self.t = 0

    def set_model(self, state_names, state_equations, parameter_names):
        self.state_names = state_names
        self.state_values.append([])
        self.state_equations = state_equations
        self.parameter_names = parameter_names

    def reset(self):
        self.t = 0
        self.state_values = []
        self.state_values.append([])
        self.predicted_values = []
        self.predicted_values.append([])


    def set_parameter_values(self, param_values):
        for p in range(len(self.parameter_names)):
            # print 'parameter ', self.parameter_names[p]
            # print ' has value: ', param_values[p]
            exec("%s = %f" % (self.parameter_names[p], param_values[p]))
            self.parameter_values.append(param_values[p])

    def set_state_values(self, state_values):
        for s in range(len(self.state_names)):
            # print 'state ', self.state_names[s],
            # print ' has value ', state_values[s]
            exec("%s = %f" % (self.state_names[s], state_values[s]))
            self.state_values[self.t].append(state_values[s])

    def print_model(self):
        for e in range(len(self.state_equations)):
            print str(self.state_names[e]) + ' = ',
            print self.state_equations[e]

    def print_model_to_file(self, file, generation):
        file.write('======================' + str(generation) + '======================\n')
        for e in range(len(self.state_equations)):
            file.write(str(self.state_names[e]) + ' = ' + str(self.state_equations[e]) + '\n')

    def to_string(self):
        result = ''
        for e in range(len(self.state_equations)):
            result += str(self.state_equations[e])
        return result

    def execute_steps(self, steps):
        for i in range(0,steps):
            self.state_values.append([0]*len(self.state_names))
            self.predicted_values.append([0]*len(self.state_names))
            self.t += 1
            for v in range(len(self.state_names)):
                value = eval(self.state_equations[v])
                if math.isinf(value) or math.isnan(value):
                    value = self.max_value
                exec("%s = %f" % (self.state_names[v], value))
                self.state_values[self.t][v] = eval(self.state_names[v])
                self.predicted_values[self.t][v] = self.state_values[self.t][v]
                # print '-----'
                # print self.t
                # print self.state_names[v]
                # print self.predicted_values[self.t][v]


    def get_values(self, state):
        if self.t == 0:
            print 'number of values ' + str(self.t)
        values = []
        index = self.state_names.index(state)
        for i in range(1, len(self.predicted_values)):
            # Limit between 0 and 1
            value = max(min(self.predicted_values[i][index], 1), 0)
            values.append(value)
        return values