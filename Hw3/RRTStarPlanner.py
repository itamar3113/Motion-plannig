import numpy as np
from RRTTree import RRTTree
import time
from tqdm import tqdm


class RRTStarPlanner(object):

    def __init__(self, bb, ext_mode, max_step_size, start, goal,
                 max_itr=None, stop_on_goal=None, k=None, goal_prob=0.01):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.step_size = max_step_size

    def sample_random_config(self, goal_prob, goal):
        if np.random.rand() < goal_prob:
            return goal
        while True:
            new_config = np.random.uniform(-np.pi, np.pi, 4)
            if self.bb.config_validity_checker(new_config):
                return new_config

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        self.tree.add_vertex(self.start)
        plan = []
        start_time = time.time()
        for i in tqdm(range(self.max_itr)):
            new_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            _, neighbor = self.tree.get_nearest_config(new_config)
            self.extend(neighbor, new_config)
            if i == 1999:
                print(f'vertices {len(self.tree.vertices)}')
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        curr_id = self.tree.get_idx_for_config(self.goal)
        if curr_id is not None:
            print(f'cost: {self.tree.vertices[curr_id].cost}')
            while curr_id != self.tree.get_idx_for_config(self.start):
                plan.append(self.tree.vertices[curr_id].config)
                curr_id = self.tree.edges[curr_id]
            plan.append(self.tree.vertices[curr_id].config)
            plan.reverse()
            return np.array(plan)
        return None

    def compute_cost(self, plan):
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost

    def rewire(self, new_id, new_config):
        neighbors_num = min(self.k, len(self.tree.vertices) - 1)
        if neighbors_num < 2:
            return
        ids, states = self.tree.get_k_nearest_neighbors(new_config, neighbors_num)
        for idx, config in zip(ids, states):
            distance = self.bb.compute_distance(new_config, config)
            state_cost = self.tree.vertices[idx].cost
            if state_cost + distance < self.tree.vertices[new_id].cost:
                if self.bb.edge_validity_checker(config, new_config):
                    self.tree.add_edge(idx, new_id, distance)

    def extend(self, near_config, rand_config):
        if self.tree.get_idx_for_config(rand_config) is None:
            if self.ext_mode == "E1":
                if self.bb.edge_validity_checker(near_config, rand_config):
                    id2 = self.tree.add_vertex(rand_config)
                    id1 = self.tree.get_idx_for_config(near_config)
                    self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, rand_config))
                    self.rewire(id2, rand_config)
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
                    self.rewire(id2, new_config)

