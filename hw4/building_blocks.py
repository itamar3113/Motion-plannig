import math
import queue
import random
import time
import threading
import multiprocessing
import timeit

import numpy as np

import environment
from environment import LocationType


def spheres_intersect(center1, radius1, center2, radius2):
    dist = np.linalg.norm(center1 - center2)
    return dist <= radius1 + radius2


class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, env, resolution=0.1, p_bias=0.2, special_bias=False):
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])
        # testing variables
        self.t_curr = 0
        self.special_bias = special_bias
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

        # self.TWO_PI = 2 * math.pi

    def sample_random_config(self, goal_prob, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if np.random.rand() < goal_prob:
            return goal_conf
        else:
            limits = list(self.env.ur_params.mechamical_limits.values())
            conf = np.zeros(len(limits))
            while True:
                for i in range(6):
                    conf[i] = np.random.uniform(limits[i][0], limits[i][1], 1)
                    if self.config_validity_checker(conf):
                        return conf

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # TODO update env? also why different from HW2?
        all_spheres = self.env.arm_transforms[self.env.active_arm].conf2sphere_coords(conf)
        for links in self.possible_link_collisions:
            spheres1 = all_spheres[links[0]]
            spheres2 = all_spheres[links[1]]
            radius1 = self.env.arm_transforms[self.env.active_arm].sphere_radius[links[0]]
            radius2 = self.env.arm_transforms[self.env.active_arm].sphere_radius[links[1]]
            for center1 in spheres1:
                for center2 in spheres2:
                    if spheres_intersect(center1, radius1, center2, radius2):
                        return True

        for link, spheres in all_spheres.items():
            for sphere in spheres:
                radius = self.env.arm_transforms[self.env.active_arm].sphere_radius[link]
                for obstacle in self.env.obstacles:
                    if spheres_intersect(sphere, radius, obstacle, self.env.radius):
                        return True
        return False

    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        num_steps = int(np.linalg.norm(prev_conf - current_conf) / self.resolution)
        num_steps = max(num_steps, 2)
        for i in range(num_steps + 1):
            conf = prev_conf + (current_conf - prev_conf) * i / num_steps
            if self.config_validity_checker(conf):
                return False
        return True

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
