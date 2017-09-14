import random
from eval.formal_method import EvaluationFramework
from model.model import Model
import math
from copy import deepcopy
from tree import Tree

class Individual():

    trees = []
    operators = {}
    terminals = {}
    number_of_equations = 0
    max_depth = 0
    param_names = []
    states = []
    operator_prob = 0.5

    def __init__(self):
        self.trees = []
        self.operators = {}
        self.terminals = {}
        self.number_of_equations = 0
        self.max_depth = 0
        self.param_names = []
        self.states = []


    def initialize(self, states, operators, terminals, max_depth, param_names):
        self.operators = operators
        self.terminals = terminals
        self.number_of_equations = len(states)
        self.states = states
        self.max_depth = max_depth
        self.param_names = param_names

        for i in range(self.number_of_equations):
            t = self.create_individual_equation(0, 'numeric')
            self.trees.append(t)

    def create_individual_equation(self, depth, required_type):
        t = Tree()
        suitable_operators = []
        # end_operators = 0
        if depth < self.max_depth:
            for operator in self.operators:
                if self.operators[operator]['output'] == required_type:
                    suitable_operators.append(operator)
                    # end_operators += 1

        suitable_terminals = []
        for terminal in self.terminals:
            if self.terminals[terminal]['output'] == required_type:
                suitable_terminals.append(terminal)
        if random.uniform(0, 1) < self.operator_prob and len(suitable_operators)>1:
            node_type = suitable_operators[int(random.uniform(0,len(suitable_operators)))]
            terminal = False
        else:
            node_type = suitable_terminals[int(random.uniform(0,len(suitable_terminals)))]
            terminal = True
        t.create_node(node_type)

        if not terminal:
            for c in self.operators[node_type]['arguments']:
                child = self.create_individual_equation(depth+1, c)
                t.add_child(child)
        return t

    def all_parameters(self, tree):
        if tree.is_terminal() and (tree.get_operator() in self.param_names):
            return True
        elif not tree.is_terminal():
            children = tree.get_children()
            for child in children:
                if not self.all_parameters(child):
                    return False
            return True
        else:
             return False

    def get_first_parameter(self, tree):
        if tree.is_terminal():
            return tree.get_operator()
        else:
            for child in tree.get_children():
                return self.get_first_parameter(child)

    def simplify_tree(self, tree):
        if not tree.is_terminal() and self.all_parameters(tree):
            tree.operator = self.get_first_parameter(tree)
            tree.children = []
        elif not tree.is_terminal():
            for child in tree.get_children():
                self.simplify_tree(child)

    def simpify_trees(self):
        for tree in self.trees:
            self.simplify_tree(tree)

    def print_trees(self):
        for i in range(len(self.trees)):
            print self.states[i] + ' = ' + self.print_tree(self.trees[i])

    def print_tree(self, tree):
        return self.print_tree_rec(tree)

    def print_tree_rec(self, tree):
        # Currently only supports binary operators and terminals
        result = ''
        if tree.is_terminal():
            result = result + tree.get_operator()
        else:
            if len(tree.get_children()) == 2:
                if self.operators[tree.get_operator()]['notation'] == 'infix':
                    result = result + '(' + self.print_tree_rec(tree.get_children()[0]) + tree.get_operator() + self.print_tree_rec(tree.get_children()[1]) + ')'
                elif self.operators[tree.get_operator()]['notation'] == 'prefix':
                    result = result + tree.get_operator() + '(' + self.print_tree_rec(tree.get_children()[0]) + ',' + self.print_tree_rec(tree.get_children()[1]) + ')'
        return result

    def parameters_used(self, equation, current_params):
        for par in self.param_names:
            if par in equation and not par in current_params:
                current_params.append(par)
        return current_params

    def return_as_model(self):
        m = Model()
        parameters_used = []
        equations = []

        for tree in self.trees:
            eq = self.print_tree(tree)
            equations.append(eq)
            parameters_used = self.parameters_used(eq, parameters_used)

        m.set_model(self.states, equations, parameters_used)
        return m