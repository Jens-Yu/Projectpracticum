#This file is used to calculate the gradient of log-likelihood
#Use SVGD to iterate the sampled points

# Import necessary modules
from manipulator import *
import numpy as np
import matplotlib.pyplot as plt
from planning import *
from utilities import *
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import mplcursors
import networkx as nx


# Define the green color map
colors1 = [(0, 1, 0, 1), (1, 1, 1, 1)]
cmap = colors.LinearSegmentedColormap.from_list('mycmap', colors1, N=256)
# Define the Gradient class
class Gradient:
    # Constructor method that takes a manipulator object as input
    def __init__(self, manipulator: manipulator):
        self.manipulator = manipulator

    # Method that returns a 2D array of norm_h_x values calculated for different values of x and y
    def get_norm_h_x_list(self, resolution=0.1):
        # Define the indices for the 2D array
        left_x_index = int(np.ceil((-np.pi) / resolution))
        left_y_index = int(np.ceil((-np.pi) / resolution))
        right_x_index = int(np.ceil((np.pi) / resolution))
        right_y_index = int(np.ceil((np.pi) / resolution))
        n = int(np.ceil(2 * np.pi / resolution))
        # Create the norm_h_x grid with the indices
        norm_h_x_grid = np.zeros([n, n])
        # Compute norm_h_x for each value of x and y and store it in the corresponding grid cell
        for i in range(left_x_index, right_x_index):
            for j in range(left_y_index, right_y_index):
                norm_h_x_grid[j+31,i+31]=self.compute_norm_h_x(i,j)
        return norm_h_x_grid

    # Method that computes the norm of h(x) for a given value of x and y
    def compute_norm_h_x(self,x_value,y_value):
        # Set the resolution and create a new manipulator object
        resolution=0.1
        ma=manipulator(dof,links)
        penal_dist=0.2
        x = x_value * resolution
        y = y_value * resolution
        # Compute the forward kinematics value for the given x and y values
        fk_val = ma.fk([x, y])
        # Compute the cost for each link in the manipulator and store it in a list
        if -np.pi <= x <= np.pi and -np.pi <= y <= np.pi:
            for k in range(ma.dof):
                cost = []
                min_dist = 10
                # Compute the minimum distance between the current link and all obstacles in the environment
                for obs in req.obstacles:
                    dist = ma.compute_distance(fk_val[k], fk_val[k + 1], ma.links_w[k], obs)
                    min_dist = min(dist, min_dist)
                # Compute the cost for the current link based on the minimum distance
                if min_dist <= penal_dist:
                    cost_links = penal_dist - min_dist
                else:
                    cost_links = 0
                cost.append(cost_links)
        else:
            cost=[0.1*x,0.1*y]
        # Compute the norm of the cost list and return its square
        norm_h_x = (np.linalg.norm(cost)) ** 2
        return norm_h_x

    # Method that returns a list of invalid nodes in the environmen
    def get_obstacle_space(self, req, contour_n=100):
        xs = np.linspace(-np.pi, np.pi, contour_n)
        ys = np.linspace(-np.pi, np.pi, contour_n)

        invalid_list = []
        nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
        for i, node in enumerate(nodes):
            is_valid = self.manipulator.check_validity(node, req.obstacles)
            if not is_valid:
                invalid_list.append(node)
        return invalid_list

    # Method that plots the obstacle space and the norm_h_x values
    def visualize_obstacle_space(self, norm_h_x_grid,req):
        # Obtain obstacle space using get_obstacle_space method
        obstacles_space = self.get_obstacle_space(req)
        # Create a figure with two subplots
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
        # Plot the obstacle space using scatter plot in the first subplot
        obst_space_x = [ns[0] for ns in obstacles_space]
        obst_space_y = [ns[1] for ns in obstacles_space]
        axs[0].scatter(obst_space_x, obst_space_y, c="r", s=1)
        # Create a contour plot using contourf function in the second subplot
        contour_n = 63
        xs = np.linspace(-np.pi, np.pi, contour_n)
        ys = np.linspace(-np.pi, np.pi, contour_n)
        xs, ys = np.meshgrid(xs, ys)
        axs[1].contourf(xs, ys, norm_h_x_grid, cmap="hot")

    #Method that calculates gradients of all norm_h_x and return a list
    def compute_gradient_vec(self, norm_h_x_grid):
        # Set the resolution to 0.1.
        resolution = 0.1
        # Calculate the number of steps needed to traverse 2π radians at the given resolution.
        n = int(np.ceil(2 * np.pi / resolution))
        # Initialize a 3-dimensional numpy array of zeros of size n x n x 2.
        grad_h_x_list = np.zeros([n, n,2])
        # Loop over the x and y dimensions of the grad_h_x_list array.
        for i in range(n-1):
            for j in range(n-1):
                # Calculate the gradient of the norm_h_x_grid at the current position.
                gradofx=(norm_h_x_grid[j+1,i]-norm_h_x_grid[j,i])/resolution
                gradofy=(norm_h_x_grid[j,i+1]-norm_h_x_grid[j,i])/resolution
                grad_h_x=(gradofx,gradofy)
                grad_h_x_list[j,i]=grad_h_x


        return grad_h_x_list

    # Method that calculates gradient of point x
    def compute_gradient(self,x,resolution=0.1):
        gradofx=(self.compute_norm_h_x((x[0]+resolution)/resolution,x[1]/resolution)-self.compute_norm_h_x(x[0]/resolution,x[1]/resolution))/resolution
        gradofy=(self.compute_norm_h_x(x[0]/resolution,(x[1]+resolution)/resolution)-self.compute_norm_h_x(x[0]/resolution,x[1]/resolution))/resolution
        gradient_vec=(-10*gradofx,-10*gradofy)

        return gradient_vec

#Function that calculates iterative step length
def compute_step(x, samp_n, datas, gradients, neg_hessian=1):
    step = 0.03
    bw =0.25
    diff = datas - x
    diff_square = np.multiply(diff, diff)
    diff_square = np.sum(diff_square, axis=1)
    kernel_product = np.exp(-0.5/bw*diff_square*neg_hessian)
    gradients = np.array(gradients)
    kernel_product = kernel_product.reshape((samp_n, 1))
    gradients = gradients.reshape((1, 2))
    first_term = np.multiply(kernel_product, gradients)
    second_term = -0.5*bw*np.multiply(diff, kernel_product)
    return np.mean(first_term+second_term, axis=0)*step, np.mean(second_term, axis=0), diff, kernel_product

#Function that push points to the nearest boundary
def push_to_boundary(point, x_min,x_max,y_min,y_max):
    x = point[0]
    y = point[1]
    if x<x_min and y>y_max:
        return [x_min,y_max]
    elif x<x_min and y_min<=y<=y_max:
        return [x_min,y]
    elif x<x_min and y<y_min:
        return [x_min,y_min]
    elif x_min<=x<=x_max and y<y_min:
        return [x,y_min]
    elif x>x_max and y<y_min:
        return [x_max,y_min]
    elif x>x_max and y_min<=y<=y_max:
        return [x_max,y]
    elif x>x_max and y>y_max:
        return [x_max,y_max]
    else:
        return [x,y_max]



#Function that uses SVGD method to iterate the sampled points
def svgd_example():
    global samp_n
    ma = manipulator(dof, links)
    gr = Gradient(ma)
    n=30
    samp_n = n * n
    x_coords = np.linspace(-np.pi, np.pi, n)
    y_coords = np.linspace(-np.pi, np.pi, n)

    xx, yy = np.meshgrid(x_coords, y_coords)

    initial_dataset = np.column_stack((xx.flatten(), yy.flatten()))


    xs = initial_dataset.tolist()
    dataset = initial_dataset
    # Iterate 100 time
    for i in range(300):
        from copy import deepcopy
        xs_copy = deepcopy(xs)
        # For each point in dataset, update the position using gradients and compute_step
        for ix, x in enumerate(xs_copy):
            gradients = gr.compute_gradient(x)
            dx, _1,_2,_3= compute_step(x,samp_n, dataset, gradients,1)
            x = x + dx
            xs[ix] = x


        dataset = np.array(xs)
    norm_h_x = gr.get_norm_h_x_list()
    obstacles_space = gr.get_obstacle_space(req)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    obst_space_x = [ns[0] for ns in obstacles_space]
    obst_space_y = [ns[1] for ns in obstacles_space]
    axs[0].scatter(obst_space_x, obst_space_y, c="r", s=1)
    contour_n = 63
    x_index = np.linspace(-np.pi, np.pi, contour_n)
    y_index = np.linspace(-np.pi, np.pi, contour_n)
    x_index, y_index = np.meshgrid(x_index, y_index)
    axs[1].contourf(x_index, y_index, norm_h_x, cmap="hot")
    axs[0].scatter(initial_dataset[:, 0], initial_dataset[:, 1], color="black")

    sc = axs[0].scatter(dataset[:, 0], dataset[:, 1])

    annot = axs[0].annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        gradient = gr.compute_gradient(pos)
        step, second_term,diff,kernel_product=compute_step(pos,samp_n,dataset,gradient,1)
        text = f"idx:({pos[0]:.2f},{pos[1]:.2f}) \n" \
               f"negative_gradient:({gradient[0]:.2f},{gradient[1]:.2f}) \n" \
               f"repulsive force:({second_term[0]:.2f},{second_term[1]:.2f}) \n"\
               f"step:({step[0]:.2f},{step[1]:.2f})\n"


        annot.set_text(text)

        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()

        if event.inaxes == axs[0]:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

def animation():
    global dataset
    global samp_n
    ma = manipulator(dof, links)
    gr = Gradient(ma)
    n = 7
    samp_n = n * n
    x_coords = np.linspace(-np.pi, np.pi, n)
    y_coords = np.linspace(-np.pi, np.pi, n)

    xx, yy = np.meshgrid(x_coords, y_coords)

    initial_dataset = np.column_stack((xx.flatten(), yy.flatten()))

    xs = initial_dataset.tolist()
    dataset = initial_dataset
    obstacles_space = gr.get_obstacle_space(req)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    obst_space_x = [ns[0] for ns in obstacles_space]
    obst_space_y = [ns[1] for ns in obstacles_space]
    ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
    ax.scatter(initial_dataset[:, 0], initial_dataset[:, 1], color="black")
    sc=ax.scatter([],[])

    def update(frame):
        global dataset
        from copy import deepcopy
        xs_copy = deepcopy(xs)
        # For each point in dataset, update the position using gradients and compute_step
        for framex, x in enumerate(xs_copy):
            gradients = gr.compute_gradient(x)
            dx, _1,_2,_3 = compute_step(x,samp_n, dataset, gradients, 1)
            x = x + dx
            xs[framex] = x

        # Update the dataset and scatter plot
        dataset = np.array(xs)
        sc.set_offsets(dataset)

        # If the animation has reached the final frame, stop the animation
        if frame == 299:
            ani.event_source.stop()

        return sc,

    # Set up the animation
    ani = FuncAnimation(fig, update, frames=range(300), interval=1, blit=True)
    plt.show()








if __name__ == "__main__":
    dof = 2
    links = [0.5, 0.5]
    ma = manipulator(dof, links)
    gr = Gradient(ma)
    request_file = "../data/pl_req/hard_pl_req_250_nodes.json"
    hard_requests = load_planning_requests(request_file)
    req = hard_requests[72]
    svgd_example()
    animation()











