import numpy as np

from RRTMotionPlanner import RRTMotionPlanner
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start
        self.not_improve_steps = 0
        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)
        self.goal = 0
        self.goal_value = 0

    def sample_random_config_rotate(self, goal_prob, goal):
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

    def sample_random_config(self, goal_prob, goal):
        coin = np.random.rand()
        if coin < goal_prob:
            new_config = goal + np.random.uniform(-np.pi / 6, np.pi / 6, 4)
            return new_config
        else:
            while True:
                new_config = np.random.uniform(-np.pi, np.pi, 4)
                if self.bb.config_validity_checker(new_config):
                    return new_config

    def expand_near(self, origin, near_id):
        valid_configs = []
        inspected_points_list = []
        distances = []
        current_points = self.tree.vertices[near_id].inspected_points
        near_config = np.copy(origin)
        new_config = near_config + np.array([0, 0, 0, np.pi / 3])
        while new_config[3] < np.pi:
            if self.bb.config_validity_checker(new_config) and self.bb.edge_validity_checker(
                    near_config, new_config):
                new_points = self.bb.compute_union_of_points(current_points, self.bb.get_inspected_points(new_config))
                valid_configs.append(np.copy(new_config))
                inspected_points_list.append(new_points)
                distances.append(self.bb.compute_distance(near_config, new_config))
                near_config = np.copy(new_config)
                current_points = new_points
                new_config = near_config + np.array([0, 0, 0, np.pi / 3])
            else:
                break
        new_config = origin - np.array([0, 0, 0, np.pi / 3])
        current_points = self.tree.vertices[near_id].inspected_points
        while new_config[3] > -np.pi:
            if self.bb.config_validity_checker(new_config) and self.bb.edge_validity_checker(
                    near_config, new_config):
                new_points = self.bb.compute_union_of_points(
                    current_points,
                    self.bb.get_inspected_points(new_config)
                )
                valid_configs.append(np.copy(new_config))
                inspected_points_list.append(new_points)
                distances.append(self.bb.compute_distance(near_config, new_config))

                # Update for next iteration
                near_config = np.copy(new_config)
                current_points = new_points
                new_config = near_config - np.array([0, 0, 0, np.pi / 3])
            else:
                break
        current_id = near_id
        for i in range(len(valid_configs)):
            new_id = self.tree.add_vertex(valid_configs[i], inspected_points_list[i])
            self.tree.add_edge(current_id, new_id, distances[i])
            current_id = new_id
        if len(valid_configs) > 0:
            new_coverage = self.bb.compute_coverage(inspected_points_list[-1])
            if new_coverage > self.coverage:
                self.coverage = new_coverage
                self.goal_value = 0
                print(self.coverage)
                self.not_improve_steps = 0
            else:
                value = len(
                    self.bb.compute_union_of_points(self.tree.vertices[self.tree.max_coverage_id].inspected_points,
                                                    inspected_points_list[-1]))
                if value > self.goal_value:
                    self.goal_value = value
                    self.goal = current_id
                    print(value)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        curr_id = self.tree.add_vertex(self.start, self.bb.get_inspected_points(self.start))
        self.coverage = self.bb.compute_coverage(self.tree.vertices[curr_id].inspected_points)
        plan = []
        start_time = time.time()
        while self.coverage < 0.75:
            new_config = self.sample_random_config(self.goal_prob, self.tree.vertices[self.tree.max_coverage_id].config)
            _, neighbor = self.tree.get_nearest_config(new_config)
            curr_id = self.extend(neighbor, new_config)
            if curr_id is not None and len(self.bb.get_inspected_points(new_config)) > 0:
                self.expand_near(new_config, curr_id)
            self.not_improve_steps += 1
            if self.not_improve_steps >= 150:
                current_inspected = self.tree.vertices[self.tree.max_coverage_id].inspected_points
                future_inspected = self.tree.vertices[self.goal].inspected_points
                if len(self.bb.compute_union_of_points(current_inspected, future_inspected)) > len(current_inspected):
                    print("rewiring...")
                    curr_id = self.rewire(self.tree.max_coverage_id, self.goal)
                self.not_improve_steps = 0
                self.goal_value = 0
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
        if self.ext_mode == "E1":
            if self.bb.config_validity_checker(rand_config) and self.bb.edge_validity_checker(near_config,
                                                                                              rand_config):
                id1 = self.tree.get_idx_for_config(near_config)
                id2 = self.tree.add_vertex(rand_config, self.bb.compute_union_of_points(
                    self.tree.vertices[id1].inspected_points, self.bb.get_inspected_points(rand_config)))
                self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, rand_config))
                return id2
        elif self.ext_mode == "E2":
            dis_to_rand = self.bb.compute_distance(near_config, rand_config)
            if dis_to_rand < self.step_size:
                new_config = rand_config
            else:
                direction = rand_config - near_config
                new_config = near_config + direction / dis_to_rand * self.step_size
            if self.bb.config_validity_checker(new_config) and self.bb.edge_validity_checker(near_config,
                                                                                             new_config):
                id1 = self.tree.get_idx_for_config(near_config)
                id2 = self.tree.add_vertex(new_config, self.bb.compute_union_of_points(
                    self.tree.vertices[id1].inspected_points, self.bb.get_inspected_points(new_config)))
                self.tree.add_edge(id1, id2, self.bb.compute_distance(near_config, new_config))
                new_coverage = self.bb.compute_coverage(self.tree.vertices[id2].inspected_points)
                if new_coverage > self.coverage:
                    self.coverage = new_coverage
                    self.goal_value = 0
                    print(self.coverage)
                    self.not_improve_steps = 0
                else:
                    value = len(self.bb.compute_union_of_points(
                        self.tree.vertices[self.tree.max_coverage_id].inspected_points,
                        self.tree.vertices[id2].inspected_points))
                    if value > self.goal_value:
                        self.goal_value = value
                        self.goal = id2
                        print(value)
                return id2
        return None

    def rewire(self, start_id, goal_id):
        curr_id = start_id
        plan_from_start = []
        plan_from_goal = []
        # get to the root from start
        while curr_id != 0:
            plan_from_start.append(curr_id)
            curr_id = self.tree.edges[curr_id]
        plan_from_start.append(curr_id)
        curr_id = goal_id
        # get the commot father and the path from goal
        while curr_id != 0:
            if curr_id in plan_from_start:
                break
            plan_from_goal.append(curr_id)
            curr_id = self.tree.edges[curr_id]
        plan_from_goal.reverse()
        prev_id = start_id
        # create from start back to common root
        for i in range(len(plan_from_start)):
            new_id = self.tree.add_vertex(self.tree.vertices[plan_from_start[i]].config,
                                          self.tree.vertices[start_id].inspected_points)
            self.tree.add_edge(prev_id, new_id)
            prev_id = new_id
            if plan_from_start[i] == curr_id:
                break
        self.tree.add_edge(prev_id, plan_from_goal[0])

        for idx in plan_from_goal:
            self.tree.vertices[idx].inspected_points = self.bb.compute_union_of_points(
                self.bb.get_inspected_points(self.tree.vertices[idx].config),
                self.tree.vertices[prev_id].inspected_points)
            prev_id = idx
        new_coverage = self.bb.compute_coverage(self.tree.vertices[goal_id].inspected_points)
        if new_coverage > self.coverage:
            self.coverage = new_coverage
            print(new_coverage)
        print('done rewiring')
        return prev_id
