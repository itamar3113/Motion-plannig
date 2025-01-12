import numpy as np


def spheres_intersect(center1, radius1, center2, radius2):
    dist = np.linalg.norm(center1 - center2)
    return dist <= radius1 + radius2


class BuildingBlocks3D(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if np.random.rand() < self.p_bias:
            return goal_conf
        else:
            limits = list(self.ur_params.mechamical_limits.values())
            conf = np.zeros(len(limits))
            for i in range(6):
                conf[i] = np.random.uniform(limits[i][0], limits[i][1], 1)
        return conf

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        all_spheres = self.transform.conf2sphere_coords(conf)
        for links in self.possible_link_collisions:
            spheres1 = all_spheres[links[0]]
            spheres2 = all_spheres[links[1]]
            radius1 = self.transform.sphere_radius[links[0]]
            radius2 = self.transform.sphere_radius[links[1]]
            for center1 in spheres1:
                for center2 in spheres2:
                    if spheres_intersect(center1, radius1, center2, radius2):
                        return True

        for link, spheres in all_spheres.items():
            for sphere in spheres:
                radius = self.transform.sphere_radius[link]
                if link != 'shoulder_link' and sphere[2] - radius < 0:
                    return True
                if sphere[0] < 0.4:  # check if from same size
                    return True
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
        print(f'num_steps: {num_steps + 1}')
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
