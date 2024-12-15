import math

import numpy as np
from twoD.environment import MapEnvironment
from twoD.building_blocks import BuildingBlocks2D
from twoD.prm import PRMController
from twoD.visualizer import Visualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR


def run_2d():
    conf = np.array([0.78, -0.78, 0.0, 0.0])

    # prepare the map
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)

    visualizer.visualize_map(config=conf)

    robot_positions = bb.compute_forward_kinematics(given_config=conf)
    print(bb.validate_robot(robot_positions=robot_positions)) # check robot validity
    print(bb.config_validity_checker(config=conf)) # check robot and map validity


def run_prm():
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    conf2 = np.array([0.8, -0.8, 0.8, 0.5])
    #conf2 = np.array([0.8, 0.8, 0.3, 0.5])

    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)
    prm = PRMController(conf1, conf2, bb)

    plan = [] # TODO: HW2 4.3.5

    print(bb.compute_path_cost(plan))
    visualizer.visualize_plan_as_gif(plan)


def generate_graph():
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    conf2 = np.array([0.8, 0.8, 0.3, 0.5])
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    prm = PRMController(conf1, conf2, bb)
    prm.create_graph()



def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          resolution=0.1,
                          p_bias=0.05, )

    env = Environment(env_idx=1, bb=bb)

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    conf1 = np.deg2rad([0, -90, 0, -90, 0, 0])

    conf2 = np.array([-0.694, -1.376, -2.212, -1.122, 1.570, -2.26])

    # ---------------------------------------

    # collision checking examples
    res = bb.is_in_collision(conf=conf1)
    res = bb.local_planner(prev_conf=conf1 ,current_conf=conf2)

    visualizer.show_conf(conf1)

if __name__ == "__main__":
    run_2d()
    # run_prm()
    # run_3d()
    # generate_graph()