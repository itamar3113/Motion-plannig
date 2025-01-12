import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def sample_random_config(self, goal_prob, goal):
        coin = np.random.rand()
        if coin < goal_prob:
            if coin < goal_prob / 2:
                if coin < goal_prob / 4:
                    new_config = goal + np.array([0, 0, 0, np.pi / 3])
                else:
                    new_config = goal - np.array([0, 0, 0, np.pi / 3])
            else:
                new_config = goal + np.random.uniform(-np.pi / 6, np.pi / 6, 4)
        else:
            new_config = np.random.uniform(-np.pi, np.pi, 4)
        return new_config

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        curr_id = self.tree.add_vertex(self.start, self.bb.get_inspected_points(self.start))
        self.coverage = self.bb.compute_coverage(self.tree.vertices[curr_id].inspected_points)
        plan = []
        start_time = time.time()
        while self.coverage < 0.5:
            new_config = self.sample_random_config(self.goal_prob, self.tree.vertices[self.tree.max_coverage_id].config)
            _, neighbor = self.tree.get_nearest_config(new_config)
            curr_id = self.extend(neighbor, new_config)
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        print(f'cost: {self.tree.vertices[curr_id].cost}')
        while curr_id != self.tree.get_idx_for_config(self.start):
            plan.append(self.tree.vertices[curr_id].config)
            curr_id = self.tree.edges[curr_id]
        plan.append(self.tree.vertices[curr_id].config)
        plan.reverse()
        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distances(plan[i], plan[i + 1])
        return cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.tree.get_idx_for_config(rand_config) is None:
            if self.ext_mode == "E1":
                if self.bb.config_validity_checker(rand_config) and self.bb.edge_validity_checker(near_config, rand_config):
                    id1 = self.tree.get_idx_for_config(near_config)
                    id2 = self.tree.add_vertex(rand_config, self.bb.compute_union_of_points(
                        self.tree.vertices[id1].inspection_points, self.bb.get_intersection_points(rand_config)))
                    self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, rand_config))
                    return id2
            elif self.ext_mode == "E2":
                dis_to_rand = self.bb.compute_distance(near_config, rand_config)
                if dis_to_rand < self.step_size:
                    new_config = rand_config
                else:
                    direction = rand_config - near_config
                    new_config = near_config + direction / dis_to_rand * self.step_size
                if self.bb.config_validity_checker(new_config) and self.bb.edge_validity_checker(near_config, new_config):
                    id1 = self.tree.get_idx_for_config(near_config)
                    id2 = self.tree.add_vertex(new_config, self.bb.compute_union_of_points(
                        self.tree.vertices[id1].inspected_points, self.bb.get_inspected_points(new_config)))
                    self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, new_config))
                    new_coverage = self.bb.compute_coverage(self.tree.vertices[id2].inspected_points)
                    if new_coverage > self.coverage:
                        self.coverage = new_coverage
                        self.goal_prob = max(new_coverage, 0.2)
                        print(self.coverage)
                    else:
                        self.goal_prob *= 99/100
                        self.goal_prob = max(0.2, self.goal_prob)
                    return id2
                else:
                    self.goal_prob *= 99 / 100
                    self.goal_prob = max(0.2, self.goal_prob)
        return None

