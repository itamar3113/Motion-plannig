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
from matplotlib import pyplot as plt
import re


def run_2d():
    conf = np.array([0.78, -0.78, 0.0, 0.0])

    # prepare the map
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)

    visualizer.visualize_map(config=conf)

    robot_positions = bb.compute_forward_kinematics(given_config=conf)
    print(bb.validate_robot(robot_positions=robot_positions))  # check robot validity
    print(bb.config_validity_checker(config=conf))  # check robot and map validity


def run_prm():
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    # conf2 = np.array([0.8, -0.8, 0.8, 0.5])
    conf2 = np.array([0.8, 0.8, 0.3, 0.5])
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)
    prm = PRMController(conf1, conf2, bb)
    visualizer.visualize_map(conf2, plan=None, show_map=True)
    plan = prm.create_graph(100, 100, 7)
    file_name = "output.txt"

    # Write the data to the file
    with open(file_name, "w") as file:
        for sublist in plan:
            file.write(", ".join(str(tup) for tup in sublist) + "\n")
    for k in plan:
        for path_data in k:
            if path_data[0] is not None:
                print(bb.compute_path_cost(path_data[0]))
                visualizer.visualize_plan_as_gif(path_data[0])


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
    env = Environment(env_idx=1)
    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          resolution=0.1,
                          p_bias=0.05, env=env)



    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    conf1 = np.deg2rad([80,-72, 101,-120,-90,-10])
    conf2 = np.deg2rad([20,-90, 90,-90,-90,-10])
    

    # ---------------------------------------

    # collision checking examples
    #res = bb.is_in_collision(conf=conf1)
    #res = bb.local_planner(prev_conf=conf1, current_conf=conf2)
    collision = True
    while collision:
        conf1 = bb.sample(conf2)
        if bb.is_in_collision(conf1):
            visualizer.show_conf(conf1)
    visualizer.show_conf(conf2)

def extract_data_from_file(file_name):
    colors = ['b', 'g', 'r', 'y', 'k']
    K = ['5', '10', 'log(n)', '10log2(n)', 'n / 10']
    n = [100, 200, 300, 400, 500, 600, 700]
    path_data = {k: {'cost': [], 'time': []} for k in K}
    # Open the file
    with open(file_name, 'r') as file:
        lines = file.readlines()
        i = 1
        k = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('cost'):
                words = line.strip().split(' ')
                path_data[K[k % 5]]['cost'].append(float(words[1].strip(',')))
                path_data[K[k % 5]]['time'].append(float(words[3].strip(',')))
            k += 1
            i += 2
    for j, k in enumerate(K):
        times = path_data[k]['time']
        gap = len(n) - len(times)
        copy_n = n
        if gap > 0:
            copy_n = n[gap:]
        plt.plot(copy_n, times, color=colors[j], label=f'k = {k}')
    plt.xlabel('number of samples')
    plt.ylabel("time (seconds)")
    plt.title('times vs number of samples')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # run_2d()
    #run_prm()
    run_3d()
    # generate_graph()

