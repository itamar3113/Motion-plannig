import numpy as np
import time

import environment
from RRTTree import RRTTree


class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)
        # testing variables
        self.t_curr = 0
        self.itr_no_goal_limit = 250
        self.sample_rotation = 0.1
        # self.TWO_PI = 2 * math.pi
        self.last_cost = -1
        self.last_ratio = 1
        self.K = 5

    def find_path(self, start_conf, goal_conf):
        self.tree.AddVertex(start_conf)
        plan = []
        start_time = time.time()
        iter_count = 0
        while iter_count < self.itr_no_goal_limit:
            new_config = self.bb.sample_random_config(self.bb.p_bias, goal_conf)
            _, neighbor = self.tree.GetNearestVertex(new_config)
            new_config = self.extend(neighbor.state, new_config)
            if new_config is not None and np.array_equal(new_config, goal_conf):
                self.tree.goal_exist = True
                break
            iter_count += 1
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        curr_id = self.tree.get_idx_for_config(goal_conf)
        if curr_id is not None:
            print(f'cost: {self.tree.vertices[curr_id].cost}')
            while curr_id != self.tree.get_idx_for_config(start_conf):
                plan.append(self.tree.vertices[curr_id].state)
                curr_id = self.tree.edges[curr_id]
            plan.append(self.tree.vertices[curr_id].state)
            plan.reverse()
            return np.array(plan), self.tree.vertices[curr_id].cost
        return None

    def rewire(self, new_id, new_config):
        neighbors_num = min(self.K, len(self.tree.vertices) - 1)
        if neighbors_num < 2:
            return
        ids, states = self.tree.GetKNN(new_config, neighbors_num)
        for idx, config in zip(ids, states):
            distance = self.bb.compute_distance(new_config, config)
            state_cost = self.tree.vertices[idx].cost
            if state_cost + distance < self.tree.vertices[new_id].cost:
                if self.bb.edge_validity_checker(config, new_config):
                    self.tree.AddEdge(idx, new_id, distance)

    def extend(self, x_near, x_random):
        '''
        Implement the Extend method
        @param x_near - Nearest Neighbor
        @param x_random - random sampled configuration
        return the extended configuration
        '''
        dis_to_rand = self.bb.compute_distance(x_near, x_random)
        if dis_to_rand < self.max_step_size:
            new_config = x_random
        else:
            direction = x_random - x_near
            new_config = x_near + direction / dis_to_rand * self.max_step_size
        if self.bb.edge_validity_checker(x_near, new_config):
            id2 = self.tree.AddVertex(new_config)
            id1 = self.tree.get_idx_for_config(x_near)
            self.tree.AddEdge(id1, id2, self.bb.compute_distance(x_near, new_config))
            self.rewire(id2, new_config)
            print(len(self.tree.vertices))
            return new_config
        return None
