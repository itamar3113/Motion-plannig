
import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import BuildingBlocks3D
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    inflation_factors = np.linspace(1.0, 1.8, 9)
    times = []
    ground_truth = []
    false_negatives = [0]
    for inflation_factor in inflation_factors:
        print("Inflation factor: ", inflation_factor)
        is_collision_instances = []
        false_negative = 0
        ur_params = UR5e_PARAMS(inflation_factor=inflation_factor)
        env = Environment(env_idx=0)
        transform = Transform(ur_params)
        bb = BuildingBlocks3D(transform=transform, ur_params=ur_params, env=env, resolution=0.1, p_bias=0.03)
        # change the path
        random_samples = np.load('random_samples_100k.npy')
        if inflation_factor == 1.0:
            start_time = time.time()
            for sample in tqdm(random_samples):
                ground_truth.append(bb.is_in_collision(sample))
            end_time = time.time()
            times.append(end_time - start_time)
        else:
            start_time = time.time()
            for sample in tqdm(random_samples):
                is_collision_instances.append(bb.is_in_collision(sample))
            end_time = time.time()
            times.append(end_time - start_time)
            for i in range(len(ground_truth)):
                if ground_truth[i] != is_collision_instances[i]:
                    false_negative += 1
            false_negatives.append(false_negative)





    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('min radii factor')
    ax2 = ax1.twinx()
    ax1.set_ylabel('time (s)', color='blue')
    ax2.set_ylabel('False Negative Instances', color='red') 
    ax1.scatter(inflation_factors, times, c='blue')
    ax2.scatter(inflation_factors, false_negatives, c='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()



