
import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import BuildingBlocks3D
import time
import matplotlib.pyplot as plt

def main():
    inflation_factors = np.linspace(1.0, 1.8, 9)
    times = []
    is_collision_instances = []
    for inflation_factor in inflation_factors:
        ur_params = UR5e_PARAMS(inflation_factor=inflation_factor)
        env = Environment(env_idx=0)
        transform = Transform(ur_params)
        bb = BuildingBlocks3D(transform=transform, ur_params=ur_params, env=env, resolution=0.1, p_bias=0.03)
        # change the path
        random_samples = np.load('./random_samples/random_samples_100k.npy')

        # TODO: HW2 5.2.5
        pass


    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('min radii factor')
    ax2 = ax1.twinx()
    ax1.set_ylabel('time (s)', color='blue')
    ax2.set_ylabel('False Negative Instances', color='red') 
    ax1.scatter(inflation_factors, times, c='blue')
    ax2.scatter(inflation_factors, is_collision_instances, c='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()



