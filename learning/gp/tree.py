import random
from eval.formal_method import EvaluationFramework
from model.model import Model
import math
from copy import deepcopy

class Tree():

    operator = []
    children = []

    def __init__(self):
        self.operator = []
        self.children = []

    def create_node(self, operator):
        self.operator = operator
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_operator(self):
        return self.operator

    def is_terminal(self):
        return (len(self.children) == 0)

    def has_depth(self):
        if len(self.children) == 0:
            return 1
        else:
            depths = []
            for c in self.children:
                depths.append(c.has_depth())
            return (1+max(depths))

    def has_number_of_nodes(self):
        if len(self.children) == 0:
            return 1
        else:
            nodes = []
            for c in self.children:
                nodes.append(c.has_number_of_nodes())
            return 1 + sum(nodes)

    def get_point(self, target_point):
        [tree, point] = self.get_point_int(1, target_point)
        return tree

    def get_point_int(self, current_point, target_point):
        if current_point == target_point:
            return self, current_point
        elif len(self.children) == 0:
            return None, current_point + 1
        else:
            new_current_point = current_point + 1
            for c in self.children:
                [tree, new_current_point] = self.get_point_int(new_current_point, target_point)
                if not tree == None:
                    return tree, new_current_point
            return None, new_current_point

    def get_subtree_depth(self, subtree):
        return self.get_subtree_depth_int(0, subtree)

    def get_subtree_depth_int(self, depth, subtree):
        if  subtree == self:
            return depth + 1
        elif len(self.children) == 0:
            return -1
        else:
            success = []
            for c in self.children:
                success.append(self.get_subtree_depth_int(self, depth+1, subtree))
            return max(success)

    def replace(self, mutation_point, new_subtree):
        point = self.replace_int(mutation_point, 1, new_subtree)

    def replace_int(self, mutation_point, current_point, new_subtree):
        if current_point == mutation_point:
            self.operator = new_subtree.operator
            self.children = new_subtree.children
            return current_point + 1
        elif len(self.children) == 0:
            return current_point + 1
        else:
            new_current_point = current_point + 1
            new_c = []
            for c in self.children:
                new_current_point = c.replace_int(mutation_point, new_current_point, new_subtree)
            return new_current_point
