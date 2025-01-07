import numpy as np
from RRTTree import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.step_size = 5

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.add_vertex(self.start)
        plan = []
        start_time = time.time()
        while not self.tree.is_goal_exists(self.goal):
            new_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            _, neighbor = self.tree.get_nearest_config(new_config)
            self.extend(neighbor, new_config)
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        curr_id = self.tree.get_idx_for_config(self.goal)
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
            cost += self.bb.compute_distances(plan[i], plan[i+1])
        return cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.tree.get_idx_for_config(rand_config) is None:
            if self.ext_mode == "E1":
                if self.bb.edge_validity_checker(near_config, rand_config):
                    id2 = self.tree.add_vertex(rand_config)
                    id1 = self.tree.get_idx_for_config(near_config)
                    self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, rand_config))
            elif self.ext_mode == "E2":
                dis_to_rand = self.bb.compute_distance(near_config, rand_config)
                if dis_to_rand < self.step_size:
                    new_config = rand_config
                else:
                    direction = rand_config - near_config
                    new_config = near_config + direction / dis_to_rand * self.step_size
                if self.bb.edge_validity_checker(near_config, new_config):
                    id2 = self.tree.add_vertex(new_config)
                    id1 = self.tree.get_idx_for_config(near_config)
                    self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, new_config))
