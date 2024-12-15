import time

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import math
import matplotlib.pyplot as plt

class PRMController:
    def __init__(self, start, goal, bb):
        self.graph = nx.Graph()
        self.bb = bb
        self.start = start
        self.goal = goal

        self.coordinates_history = []
        # Feel free to add class variables as you wish

    def run_PRM(self, num_coords=100, k=5):
        """
            find a plan to get from current config to destination
            return-the found plan and None if couldn't find
        """
        path = []
        # TODO: HW2 4.3.4
        # Preprocessing

        # Planning part

    def create_graph(self, base_number, how_many_to_add, num_searches):
        # TODO: HW2 4.3.5
        pass

    def gen_coords(self, n=5):
        """
        Generate 'n' random collision-free samples called milestones.
        n: number of collision-free configurations to generate
        """
        # TODO: HW2 4.3.1
        pass

    def add_to_graph(self, configs, k):
        """
            add new configs to the graph.
        """
        # TODO: HW2 4.3.2
        pass

    def find_nearest_neighbour(self, config, k=5):
        """
            Find the k nearest neighbours to config
        """
        # TODO: HW2 4.3.2
        pass

    def shortest_path(self):
        """
            Find the shortest path from start to goal using Dijkstra's algorithm (you can use previous implementation from HW1)'
        """
        # TODO: HW2 4.3.3
        pass