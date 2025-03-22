import json
import time
from enum import Enum
import numpy as np
import params
from environment import Environment

from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from inverse_kinematics import DH_matrix_UR5e, inverse_kinematic_solution, forward_kinematic_solution
from environment import LocationType


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + 'output.txt', 'a') as file:
        file.write(f"{written_log}\n")


class Gripper(str, Enum):
    OPEN = "OPEN",
    CLOSE = "CLOSE"
    STAY = "STAY"


def get_shifted_cubes_to_real_world(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env):
    cubes_already_moved = []
    cubes_in_original_area = []
    for cube in cubes_in_original_area_pre_shift:
        cubes_in_original_area.append(cube + env.cube_area_corner[LocationType.RIGHT])
    for cube in cubes_already_moved_pre_shift:
        cubes_already_moved.append(cube + env.cube_area_corner[LocationType.LEFT])
    return [*cubes_already_moved, *cubes_in_original_area]


def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0  # deviation from the first link 0 angle
        self.left_arm_base_delta = 0  # deviation from the first link 0 angle
        self.right_arm_meeting_safety = None
        self.left_arm_meeting_safety = None

        # tunable params
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1
        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        # result dict
        self.experiment_result = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes,
                                                     gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf,
                        cubes_real,
                        gripper_pre, gripper_post):
        path, cost = planner.find_path(start_conf=start_conf,
                                       goal_conf=goal_conf)  # add manipulator as argument?
        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(description,
                                                          active_id,
                                                          command,
                                                          static_arm_conf.tolist(),
                                                          [path_element.tolist() for path_element in path],
                                                          [list(cube) for cube in cubes_real],
                                                          gripper_pre,
                                                          gripper_post)
        return path

    def plan_single_cube_passing(self, cube_i, cubes,
                                 left_arm_start, right_arm_start,
                                 env, bb, planner, left_arm_transform, right_arm_transform, ):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],  # text to be displayed on the animation
            "active_id": [],  # active arm id
            "command": [],
            "static": [],  # static arm conf
            "path": [],  # active arm path
            "cubes": [],  # coordinates of cubes on the board at the given timestep
            "gripper_pre": [],  # active arm pre path gripper action (OPEN/CLOSE/STAY)
            "gripper_post": []  # active arm pre path gripper action (OPEN/CLOSE/STAY)
        }
        self.experiment_result.append(single_cube_passing_info)
        ###############################################################################

        # set variables
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
        # fix obstacles and update env
        cubes_already_moved_pre_shift = cubes[0:cube_i]
        cubes_in_original_area_pre_shift = cubes[cube_i:]
        cubes_real = get_shifted_cubes_to_real_world(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift,
                                                     env)

        update_environment(env, active_arm, left_arm_start, cubes_real)
        pickup_location = np.array(cubes_real[cube_i]) - np.array(env.arm_base_location[active_arm]) + np.array(
            [0, 0, params.before_pickup_height])
        pickup_transform = self.transformation_matrix(pickup_location, params.pickup_angles)
        pickup_iks = inverse_kinematic_solution(DH_matrix_UR5e, pickup_transform)
        cube_approach = self.sol_from_ik(pickup_iks, bb, pickup_location)
        # plan the path
        r_home_to_cube = self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm,
                                              "move",
                                              left_arm_start, cubes_real, Gripper.OPEN, Gripper.STAY)
        ###############################################################################

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(left_arm_start),
                                                          [0, 0, -params.pickup_lowering],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go up",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(left_arm_start),
                                                          [0, 0, params.pickup_lowering],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        del cubes_real[cube_i]
        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)

        self.push_step_info_into_single_cube_passing_data("right_arm => [cube pickup -> start], left_arm static",
                                                          active_arm,
                                                          'move',
                                                          list(left_arm_start),
                                                          [path_element.tolist() for path_element in
                                                           reversed(r_home_to_cube)],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        r_home_to_meeting = self.plan_single_arm(planner, self.right_arm_home, self.right_arm_meeting_safety,
                                                 "right_arm => [cube pickup -> meeting point], left_arm static",
                                                 active_arm, "move",
                                                 left_arm_start, cubes_real, Gripper.STAY, Gripper.STAY)

        active_arm = LocationType.LEFT
        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)

        l_home_to_meeting = self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_safety,
                                                 "left_arm => [home -> meeting point], right_arm static", active_arm,
                                                 'move',
                                                 self.right_arm_meeting_safety, cubes_real, Gripper.STAY, Gripper.OPEN)

        self.push_step_info_into_single_cube_passing_data(
            "Left arm grip", active_arm, "movel",
            self.right_arm_meeting_safety.tolist(), [params.close_meeting_gap, 0, 0],
            [list(c) for c in cubes_real],
            Gripper.STAY, Gripper.CLOSE
        )

        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)

        self.push_step_info_into_single_cube_passing_data(
            "Right arm release", LocationType.RIGHT, "movel",
            self.left_arm_meeting_safety.tolist(), [0, 0, 0],
            [list(c) for c in cubes_real],
            Gripper.STAY, Gripper.OPEN
        )

        description = "left_arm => [meeting point -> Zone B], right_arm static"
        active_arm = LocationType.LEFT
        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)

        offset = (1 + cube_i) * params.offset_factor
        putting_position = np.array(
            [env.cube_area_corner[LocationType.LEFT][0] + offset, env.cube_area_corner[active_arm][1] + 0.1, 0])
        putting_position = putting_position - np.array(env.arm_base_location[active_arm]) + np.array(
            [0, 0, params.before_pickup_height])

        putting_transform = self.transformation_matrix(putting_position, params.pickup_angles)
        putting_IKS = inverse_kinematic_solution(DH_matrix_UR5e, putting_transform)
        putting_conf = self.sol_from_ik(putting_IKS, bb, putting_position)

        l_meeting_to_cube = self.plan_single_arm(planner, self.left_arm_meeting_safety, putting_conf, description,
                                                 active_arm,
                                                 "move",
                                                 self.right_arm_meeting_safety, cubes_real, Gripper.STAY, Gripper.STAY)

        self.push_step_info_into_single_cube_passing_data(
            "left_arm => [placing cube], right_arm static",
            active_arm,
            "movel",
            self.right_arm_meeting_safety.tolist(),
            [0, 0, -params.pickup_lowering],
            [list(cube) for cube in cubes_real],
            Gripper.STAY,
            Gripper.OPEN
        )

        cubes[cube_i] = [offset, 0.1, 0.02]

        self.push_step_info_into_single_cube_passing_data(
            "left_arm => [moving up], right_arm static",
            active_arm,
            "movel",
            self.right_arm_meeting_safety.tolist(),
            [0, 0, params.pickup_lowering],  # Move up
            [list(cube) for cube in cubes_real],
            Gripper.STAY,
            Gripper.STAY
        )

        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)

        self.push_step_info_into_single_cube_passing_data("left_arm => [meeting], right_arm static",
                                                          active_arm,
                                                          'move',
                                                          list(self.right_arm_meeting_safety),
                                                          [path_element.tolist() for path_element in
                                                           reversed(l_meeting_to_cube)],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        self.push_step_info_into_single_cube_passing_data("right_arm => [meeting -> start], left_arm static",
                                                          LocationType.RIGHT,
                                                          'move',
                                                          list(self.left_arm_meeting_safety),
                                                          [path_element.tolist() for path_element in
                                                           reversed(r_home_to_meeting)],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.STAY)
        return self.left_arm_meeting_safety, self.right_arm_home

    @staticmethod
    def transformation_matrix(location, angles):
        x, y, z = location
        alpha, beta, gamma = angles
        return np.matrix([
            [np.cos(beta) * np.cos(gamma), np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),
             np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma), x]
            ,
            [np.cos(beta) * np.sin(gamma), np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
             np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma), y]
            ,
            [-np.sin(beta), np.sin(alpha) * np.cos(beta), np.cos(alpha) * np.cos(beta), z]
            ,
            [0, 0, 0, 1]
        ])

    @staticmethod
    def sol_from_ik(ik, bb, location):
        candidate_sols = []
        for i in range(ik.shape[1]):
            candidate_sols.append(ik[:, i])
        candidate_sols = np.array(candidate_sols)
        # check for collisions and angles limits
        sols = []
        for candidate_sol in candidate_sols:
            if bb.config_validity_checker(candidate_sol):
                continue
            for idx, angle in enumerate(candidate_sol):
                if 2 * np.pi > angle > np.pi:
                    candidate_sol[idx] = -(2 * np.pi - angle)
                if -2 * np.pi < angle < -np.pi:
                    candidate_sol[idx] = -(2 * np.pi + angle)
            if np.max(candidate_sol) > np.pi or np.min(
                    candidate_sol) < -np.pi:
                continue
            sols.append(candidate_sol)
        # verify solution:
        min_diff = 0.05
        final_sol = []
        for sol in sols:
            transform = forward_kinematic_solution(
                DH_matrix_UR5e, sol)
            diff = np.linalg.norm(np.array([transform[0, 3],
                                            transform[1, 3], transform[2, 3]]) - np.array(location))
            if diff < min_diff:
                min_diff = diff
                final_sol = sol
        final_sol = np.array(final_sol).flatten()
        return final_sol

    def plan_experiment(self):
        start_time = time.time()

        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        middle_point_diff = (np.array(env.arm_base_location[LocationType.RIGHT]) - np.array(
            env.arm_base_location[LocationType.LEFT])) / 2
        right_location = np.array([0, 0, params.meeting_height]) - middle_point_diff
        left_location = np.array([-params.meeting_gap, 0, params.meeting_height]) + middle_point_diff

        right_transform = self.transformation_matrix(right_location, params.right_angles)
        left_transform = self.transformation_matrix(left_location, params.left_angles)

        right_IKS = inverse_kinematic_solution(DH_matrix_UR5e, right_transform)
        left_IKS = inverse_kinematic_solution(DH_matrix_UR5e, left_transform)

        exp_id = 2

        transform_right_arm = Transform(ur_params=ur_params_right,
                                        ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        bb = Building_Blocks(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias, )

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id)
        log(msg="calculate meeting point for the test.")
        update_environment(env, LocationType.RIGHT, self.left_arm_home,
                           get_shifted_cubes_to_real_world(self.cubes, [], env))
        self.right_arm_meeting_safety = self.sol_from_ik(right_IKS, bb, right_location)
        update_environment(env, LocationType.LEFT, self.right_arm_meeting_safety,
                           get_shifted_cubes_to_real_world(self.cubes, [], env))
        self.left_arm_meeting_safety = self.sol_from_ik(left_IKS, bb, left_location)

        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start,
                                                                            right_arm_start,
                                                                            env, bb, rrt_star_planner, left_arm_start,
                                                                            right_arm_start)

        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")
        # Serializing json
        json_object = json.dumps(self.experiment_result, indent=4)
        # Writing to sample.json
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)
        # show the experiment then export it to a GIF
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id):
        cube_side = 0.04
        cubes = []
        if experiment_id == 1:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        elif experiment_id == 2:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            # row 1: cube 2
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        return cubes
