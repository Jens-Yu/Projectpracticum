# Name: Jiaming Yu
# Time:
import numpy as np
import matplotlib.pyplot as plt

x_coords = np.linspace(-np.pi, np.pi, 10)
y_coords = np.linspace(-np.pi, np.pi, 10)

xx, yy = np.meshgrid(x_coords, y_coords)

initial_dataset = np.column_stack((xx.flatten(), yy.flatten()))
y_0 = np.zeros_like(initial_dataset[:, 0])

ndg = normal_distribution_gradient_2d(0, 1)

xs = initial_dataset.tolist()
dataset = initial_dataset
for i in range(100):
    gradients = ndg.get_gradient_prob(dataset)
    for ix, x in enumerate(xs):
        hessian = ndg.get_hessian_prob(x)
        x = x + compute_step_2d(x, dataset, gradients, hessian, 1)
        xs[ix] = x
    dataset = np.array(xs)

plt.scatter(initial_dataset[:, 0], initial_dataset[:, 1], color="red")
plt.scatter(dataset[:, 0], dataset[:, 1])

plt.xlim([-np.pi, 2 * np.pi])
plt.ylim([-np.pi, 2 * np.pi])
plt.show()
