import itertools

import numpy as np
from shapely.geometry import Point, LineString

class BuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # define robot properties
        self.links = np.array([80.0, 70.0, 40.0, 40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        return np.linalg.norm(prev_config - next_config)

    def compute_path_cost(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += self.compute_distance(path[i], path[i + 1])
        return total_cost

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        # positions are 2D points + angle (3 dimensions) for each of the links of the robot
        cumulative_angles = np.cumsum(given_config)
        cos_angles = np.cos(cumulative_angles)
        sin_angles = np.sin(cumulative_angles)
        x = self.env.xlimit[0] + np.cumsum(self.links * cos_angles)
        y = self.env.ylimit[0] + np.cumsum(self.links * sin_angles)
        return np.column_stack((x, y))

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1, len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2 * np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2 * np.pi
        else:
            return link_angle + given_angle

    @staticmethod
    def segments_intersect(seg1: np.ndarray, seg2: np.ndarray) -> bool:
        p1, p2 = seg1
        p3, p4 = seg2

        # Computing the orientation of ordered triplets
        def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else 2

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        return (o1 != o2) and (o3 != o4)

    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        segments = np.array([
            [robot_positions[i], robot_positions[i + 1]]
            for i in range(len(robot_positions) - 1)
        ])
        for i in range(len(segments) - 2):
            for j in range(i + 2, len(segments)):
                if self.segments_intersect(segments[i], segments[j]):
                    return False
        return True

    def config_validity_checker(self, config):
        '''
        Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
        Return false if the config is not applicable, and true otherwise.
        @param config The given configuration of the robot.
        '''
        # compute robot links positions
        robot_positions = self.compute_forward_kinematics(given_config=config)

        # add position of robot placement ([0,0] - position of the first joint)
        robot_positions = np.concatenate([np.zeros((1, 2)), robot_positions])

        # verify that the robot do not collide with itself
        if not self.validate_robot(robot_positions=robot_positions):
            return False

        # verify that all robot joints (and links) are between world boundaries
        non_applicable_poses = [
            (x[0] < self.env.xlimit[0] or x[1] < self.env.ylimit[0] or x[0] > self.env.xlimit[1] or x[
                1] > self.env.ylimit[1]) for x in robot_positions]
        if any(non_applicable_poses):
            return False

        # verify that all robot links do not collide with obstacle edges
        # for each obstacle, check collision with each of the robot links
        robot_links = [LineString([Point(x[0], x[1]), Point(y[0], y[1])]) for x, y in
                       zip(robot_positions.tolist()[:-1], robot_positions.tolist()[1:])]
        for obstacle_edges in self.env.obstacles_edges:
            for robot_link in robot_links:
                obstacle_collisions = [robot_link.crosses(x) for x in obstacle_edges]
                if any(obstacle_collisions):
                    return False

        return True

    def edge_validity_checker(self, config1, config2):
        '''
        A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
        that the links during motion do not collide with anything.
        @param config1 The source configuration of the robot.
        @param config2 The destination configuration of the robot.
        '''
        # interpolate between first config and second config to verify that there is no collision during the motion
        required_diff = 0.05
        interpolation_steps = int(np.linalg.norm(config2 - config1) // required_diff)
        if interpolation_steps > 0:
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)

            # compute robot links positions for interpolated configs
            configs_positions = np.apply_along_axis(self.compute_forward_kinematics, 1, interpolated_configs)

            # compute edges between joints to verify that the motion between two configs does not collide with anything
            edges_between_positions = []
            for j in range(self.dim):
                for i in range(interpolation_steps - 1):
                    edges_between_positions.append(LineString(
                        [Point(configs_positions[i, j, 0], configs_positions[i, j, 1]),
                         Point(configs_positions[i + 1, j, 0], configs_positions[i + 1, j, 1])]))

            # check collision for each edge between joints and each obstacle
            for edge_pos in edges_between_positions:
                for obstacle_edges in self.env.obstacles_edges:
                    obstacle_collisions = [edge_pos.crosses(x) for x in obstacle_edges]
                    if any(obstacle_collisions):
                        return False

            # add position of robot placement ([0,0] - position of the first joint)
            configs_positions = np.concatenate([np.zeros((len(configs_positions), 1, 2)), configs_positions], axis=1)

            # verify that the robot do not collide with itself during motion
            for config_positions in configs_positions:
                if not self.validate_robot(config_positions):
                    return False

            # verify that all robot joints (and links) are between world boundaries
            if len(np.where(configs_positions[:, :, 0] < self.env.xlimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] < self.env.ylimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 0] > self.env.xlimit[1])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] > self.env.ylimit[1])[0]) > 0:
                return False
        return True

    def get_inspected_points(self, config):
        '''
        A function to compute the set of points that are visible to the robot with the given configuration.
        The function will return the set of points that is visible in terms of distance and field of view (FOV) and are not hidden by any obstacle.
        @param config The given configuration of the robot.
        '''
        # get robot end-effector position and orientation for point of view
        ee_pos = self.compute_forward_kinematics(given_config=config)[-1]
        ee_angle = self.compute_ee_angle(given_config=config)

        # define angle range for the ee given its position and field of view (FOV)
        ee_angle_range = np.array([ee_angle - self.ee_fov / 2, ee_angle + self.ee_fov / 2])

        # iterate over all inspection points to find which of them are currently inspected
        inspected_points = np.array([])
        for inspection_point in self.env.inspection_points:

            # compute angle of inspection point w.r.t. position of ee
            relative_inspection_point = inspection_point - ee_pos
            inspection_point_angle = self.compute_angle_of_vector(vec=relative_inspection_point)

            # check that the point is potentially visible with the distance from the end-effector
            if np.linalg.norm(relative_inspection_point) <= self.vis_dist:

                # if the resulted angle is between the angle range of the ee, verify that there are no interfering obstacles
                if self.check_if_angle_in_range(angle=inspection_point_angle, ee_range=ee_angle_range):

                    # define the segment between the inspection point and the ee
                    ee_to_inspection_point = LineString(
                        [Point(ee_pos[0], ee_pos[1]), Point(inspection_point[0], inspection_point[1])])

                    # check if there are any collisions of the vector with some obstacle edge
                    inspection_point_hidden = False
                    for obstacle_edges in self.env.obstacles_edges:
                        for obstacle_edge in obstacle_edges:
                            if ee_to_inspection_point.intersects(obstacle_edge):
                                inspection_point_hidden = True

                    # if inspection point is not hidden by any obstacle, add it to the visible inspection points
                    if not inspection_point_hidden:
                        if len(inspected_points) == 0:
                            inspected_points = np.array([inspection_point])
                        else:
                            inspected_points = np.concatenate([inspected_points, [inspection_point]], axis=0)

        return inspected_points

    def compute_angle_of_vector(self, vec):
        '''
        A utility function to compute the angle of the vector from the end-effector to a point.
        @param vec Vector from the end-effector to a point.
        '''
        vec = vec / np.linalg.norm(vec)
        if vec[1] > 0:
            return np.arccos(vec[0])
        else:  # vec[1] <= 0
            return -np.arccos(vec[0])

    def check_if_angle_in_range(self, angle, ee_range):
        '''
        A utility function to check if an inspection point is inside the FOV of the end-effector.
        @param angle The angle beteen the point and the end-effector.
        @param ee_range The FOV of the end-effector.
        '''
        # ee range is in the expected order
        if abs((ee_range[1] - self.ee_fov) - ee_range[0]) < 1e-5:
            if angle < ee_range.min() or angle > ee_range.max():
                return False
        # ee range reached the point in which pi becomes -pi
        else:
            if angle > ee_range.min() or angle < ee_range.max():
                return False

        return True

    def compute_union_of_points(self, points1, points2):
        '''
        Compute a union of two sets of inpection points.
        @param points1 list of inspected points.
        @param points2 list of inspected points.
        '''
        if points1.size == 0:
            return points2
        if points2.size == 0:
            return points1
        combined_points = np.vstack((points1, points2))
        unique_points = np.unique(combined_points, axis=0)
        return unique_points

    def compute_coverage(self, inspected_points):
        '''
        Compute the coverage of the map as the portion of points that were already inspected.
        @param inspected_points list of inspected points.
        '''
        return len(inspected_points) / len(self.env.inspection_points)
