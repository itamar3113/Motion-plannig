from matplotlib import pyplot as plt


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
