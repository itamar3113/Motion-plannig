from matplotlib import pyplot as plt
import re

epsilons = [0.2, 0.1, 0.01]
dimensions = list(range(2, 11))
colors = ['b', 'g', 'r']
for i, e in enumerate(epsilons):
    eta = [(1-(1-e)**d) for d in dimensions]
    plt.plot(dimensions, eta, color=colors[i], label=f"epsilon={e}")
plt.xlabel('dimension')
plt.ylabel("eta_d(epsilon)")
plt.legend()
plt.show()

colors = ['b', 'g', 'r', 'y', 'g']
K = ['5', '10', 'log(n)', '10log2(n)', 'n / 10']
n = [100, 200, 300, 400, 500, 600, 700]
k5_dis = [18.0893, 22.876, ]
k5_price = [14.5638, 10.5347, ]


