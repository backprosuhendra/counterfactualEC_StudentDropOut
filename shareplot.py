import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
x0 = np.arange(20)
y0 = np.random.rand(20,)

x1 = np.arange(20)
y1 = np.random.rand(20,)

x = np.random.rand(20,)
y = np.random.rand(20,) 
objective = np.column_stack((x,y))
# print(objective)

isPareto = np.ones(len(objective),dtype=bool)
# print(isPareto,"\n")

for i, point in enumerate(objective):
    dominate = np.any(
        np.all(objective <= point, axis=1) &
        np.any(objective < point, axis=1)
    )
    isPareto[i]=not dominate
paretoPoints = objective[isPareto]
paretoPoints = paretoPoints[
    np.argsort(paretoPoints[:,0])
]
# print(" ")
# print(dominate)
# print(isPareto)
# print(paretoPoints)
# plt.figure(figsize=(9,5))
# plt.scatter(x,y)
# plt.title("trade-offs counterfactual")
# plt.xlabel('validity')
# plt.ylabel('proximity')

# plt.plot(
#     paretoPoints[:,0],
#     paretoPoints[:,1],
#     'r-o',
#     linewidth=2,
#     label = "pareto front"
# )
# plt.show()
# Plot garis Pareto front
# Plot semua kandidat

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
axes[0].plot(x0,y0)
axes[0].set_title("Target Loss")
axes[1].plot(x1,y1)
axes[1].set_title("proximity")
plt.show()
plt.scatter(
    x,
    y,
    alpha=0.4,
    label="All candidates"
)

# Plot garis Pareto front
plt.plot(
    paretoPoints[:, 0],
    paretoPoints[:, 1],
    'r-o',
    linewidth=2,
    label="Pareto front"
)

plt.xlabel("Validity Loss")
plt.ylabel("Proximity Distance")
plt.title("Pareto Front: Validity vs Proximity")
plt.legend()
plt.grid(alpha=0.3)
plt.show()