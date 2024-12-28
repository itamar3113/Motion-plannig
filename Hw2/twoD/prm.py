import time
import random
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import math
import matplotlib.pyplot as plt
import heapq


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
        # Preprocessing
        # Planning part
        path_nodes, _ = self.shortest_path()
        if path_nodes is not None:
            for node in path_nodes:
                path.append(self.graph.nodes[node]['pos'])
            return path
        return None

    def create_graph(self, base_number, how_many_to_add, num_searches):
        res = {}
        path = None
        configs = self.gen_coords(base_number * num_searches)
        for i in range(num_searches):
            n = base_number + i * how_many_to_add
            K = [5, 10, math.log2(n), 10 * math.log2(n), n/10]
            for k in K:
                self.graph.clear_edges()
                self.add_to_graph(configs[:n], k)
                path = self.run_PRM(n, k)
            if path is not None:
                return path
        return path

    def gen_coords(self, n=5):
        """
        Generate 'n' random collision-free samples called milestones.
        n: number of collision-free configurations to generate
        """
        samples = []
        while len(samples) < n:
            new_config = np.random.uniform(-math.pi, math.pi, 4)
            if self.bb.config_validity_checker(new_config):
                samples.append(new_config)
        return samples

    def add_to_graph(self, configs, k):
        """
            add new configs to the graph.
        """
        self.graph.add_node('start', pos=self.start)
        self.graph.add_node('goal', pos=self.goal)
        for i, config in enumerate(configs):
            self.graph.add_node(str(i), pos=config)
        for i, node in enumerate(self.graph.nodes()):
            nn = self.find_nearest_neighbour(self.graph.nodes[node]['pos'], k)
            for neighbor in nn:
                if self.bb.edge_validity_checker(self.graph.nodes[node]['pos'], self.graph.nodes[neighbor]['pos']):
                    self.graph.add_edge(node, neighbor, weight=self.bb.compute_distance(self.graph.nodes[node]['pos'], self.graph.nodes[neighbor]['pos']))

    def find_nearest_neighbour(self, config, k=5):
        """
            Find the k nearest neighbours to config
        """
        positions = {node: data['pos'] for node, data in self.graph.nodes(data=True)}
        coords = list(positions.values())
        nodes = list(positions.keys())
        tree = KDTree(coords)
        distances, indices = tree.query(config, k=k + 1)
        res = [nodes[i] for i in indices[1:]]
        return res

    def shortest_path(self):
        """
            Find the shortest path from start to goal using Dijkstra's algorithm (you can use previous implementation from HW1)'
        """
        heap = [(0, 'start')]
        distances = {vertex: float('inf') for vertex in self.graph.nodes()}
        fathers = {vertex: None for vertex in self.graph.nodes()}
        distances['start'] = 0
        visited = set()
        goal_reached = False
        while heap:
            current_distance, vertex = heapq.heappop(heap)
            if vertex in visited:
                continue
            if vertex == 'goal':
                goal_reached = True
                break
            visited.add(vertex)
            for neighbor, weight in self.graph[vertex].items():
                distance = current_distance + weight['weight']
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    fathers[neighbor] = vertex
                    heapq.heappush(heap, (distance, neighbor))
        if goal_reached:
            stack = ['goal']
            while stack[-1] != 'start':
                stack.append(fathers[stack[-1]])
            path = []
            while stack:
                path.append(stack.pop())
            return path, distances['goal']
        return None, None
