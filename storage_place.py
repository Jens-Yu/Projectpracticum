import grp

import numpy as np
import matplotlib.pyplot as plt
from generate_classification_dataset import *
from model import FEST, GMM
import torch
import torch.distributions as D
from utilities import *


class normal_distribution_gradient:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2

    def get_gradient_log_prob(self, x):
        return -1/self.sigma2*(x - self.mu)

    def get_hessian_log_prob(self, x):
        return 1/self.sigma2

    def get_gradient_prob(self, x):
        gradients = -1/self.sigma2*(x - self.mu) * np.exp(-0.5*(x-self.mu)*(x-self.mu)/self.sigma2)
        noise = np.random.randn(np.shape(gradients)[0])/5
        return gradients + noise

    def get_hessian_prob(self, x):
        exponential = np.exp(-0.5*(x-self.mu)*(x-self.mu)/self.sigma2)
        first_term = -1/self.sigma2 * exponential
        second_term = (-1/self.sigma2*(x - self.mu))**2*exponential
        return first_term + second_term

#Function that calculates iterative step length
def compute_step(x, datas, gradients, neg_hessian=1):
    step = 0.02
    bw = 0.5
    diff = datas - x
    diff_square = np.exp(-0.5/bw*np.multiply(diff, diff)*neg_hessian)
    first_term = np.multiply(diff_square, gradients)
    second_term = -0.5/bw*np.multiply(datas, diff_square)
    return np.mean(first_term + second_term)*step


def svgd_example():
    initial_dataset = np.linspace(-2, 0, 30)
    y_0 = np.zeros_like(initial_dataset)

    ndg = normal_distribution_gradient(0, 1)

    xs = initial_dataset.tolist()
    dataset = initial_dataset
    for i in range(100):
        gradients = ndg.get_gradient_prob(dataset)
        for ix, x in enumerate(xs):
            hessian = ndg.get_hessian_prob(x)
            x = x + compute_step(x, dataset, gradients, 1)
            xs[ix] = x
        dataset = np.array(xs)

    plt.scatter(initial_dataset.tolist(), y_0 + 1, color="red")
    plt.scatter(dataset.tolist(), y_0)

    plt.show()


def process_data(config):
    # Get node usages
    # - Planning through all the planning request
    # - record which one are used for which planning request
    run_record_node_usages = config.get("record_node_usages", True)
    run_record_original_path = config.get("run_record_original_path", True)
    run_process_original_path = config.get("run_process_original_path", True)
    run_record_blocked_position_and_plan_bs = config.get("run_record_blocked_position_and_plan_bs", True)

    node_usage_file = config["node_usage_file"]
    original_path_file = config["original_path_file"]
    processed_path_file = config["processed_path_file"]
    processed_plan_b_file = config["processed_plan_b_file"]
    training_config_name = config["training_config"]

    # Record node usages given a graph and requests
    if run_record_node_usages:
        print("----------------------------------------------")
        print("Running record_node_usage...")
        record_node_usage(config)

    # Get paths with and without obstacles and save
    if run_process_original_path:
        print("----------------------------------------------")
        print("Running record_original_path...")
        record_original_path(config)

    # Analyze the planning results - get the following information
    # {"request": node_request_map[node_id], "planB_nodes", "planB_node_states"
    # Read: original_path_file - Write: processed_path_file
    if run_process_original_path:
        print("----------------------------------------------")
        print("Running process_original_path_data_file...")
        process_original_path_data_file(original_path_file, processed_path_file)

    # Get exact blocked nodes and the corresponding plan-Bs
    if run_record_blocked_position_and_plan_bs:
        print("----------------------------------------------")
        print("Running record_blocked_position_and_plan_bs...")
        record_blocked_position_and_plan_bs(config)

    # Prepare config for training
    gmm_training_config = {"data_file": [processed_plan_b_file, processed_path_file],
                           "graph_file": ""}

    fest_training_config = {"original_path_file": original_path_file}

    # Tips for training
    print("Now you can run the file train_GMM.py with proper configuration file to train a GMM model!")


def visualize_result(config):
    # Print the planning results

    # Visualize the value with and without obstacles

    # Visualize selected planning request

    # Visualize selected planning results and important nodes

    # Save all planning results as images

    pass


def map_static_blocked_region_to_plan_b(config, pl_env: Planning, req: PlanningRequest):
    """
    Get the blocked region blocked by static obstacles,
    compute their plan_Bs and visualize them weighted by the estimated values
    Figure 1:
    - Estimated value with obstacles & plan B model & statically blocked nodes

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get config
    gmm_model_file = config["gmm_model_file"]
    fest_model_file = config["fest_model_file"]
    k_gmm = config["k_gmm"]

    fest_model = FEST().double()
    gmm_model = GMM(k_gmm).double()

    fest_model.load_state_dict(torch.load(fest_model_file, map_location=device))
    gmm_model.load_state_dict(torch.load(gmm_model_file, map_location=device))

    # Get the blocked region
    colliding_samples = pl_env.get_obstacle_space(req, contour_n=10)
    colliding_samples = [torch.from_numpy(sample) for sample in colliding_samples]
    colliding_samples_tensor = torch.stack(colliding_samples).double().to(device)

    # Get the value estimations
    fest_values = fest_model(colliding_samples_tensor)
    fest_values = fest_values / torch.sum(fest_values)
    print(fest_values, torch.sum(fest_values))

    # Set the contour points
    contour_n = 80
    xs = np.linspace(-np.pi, np.pi, contour_n)
    ys = np.linspace(-np.pi, np.pi, contour_n + 1)

    nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    nodes_tensor = torch.stack(nodes_tensor).to(device)

    # Get the plan-B distributions
    # Ensure the pytorch return gradients with respect to the gradients
    mean, log_std, co_eff, log_w = gmm_model(colliding_samples_tensor)

    # Mean shape: [n_data, k_gmm*2]
    n_samples = len(colliding_samples)
    mean = mean.reshape([n_samples, k_gmm, -1])
    sigma = log_std.reshape([n_samples, k_gmm, -1]).exp()
    co_eff = co_eff.reshape([n_samples, k_gmm, -1])

    # compute the weight
    mixture_list = []
    log_prob_collection = torch.zeros(
        [n_samples, contour_n, contour_n + 1]).to(device)
    for i in range(n_samples):
        cov_mat_list = form_cov_matrix(sigma[i], co_eff[i], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[i], scale_tril=cov_mat_tensor), 0)  # Not sure about 1
        w = D.Categorical(log_w[i].squeeze().exp())

        # Mixture model given the node i
        mixture = D.MixtureSameFamily(w, dist)
        mixture_list.append(mixture)

        # Compute the log probabilities
        log_prob = mixture.log_prob(nodes_tensor)

        # Compute the weighted log_prob
        print("log_prob_collection shape", log_prob_collection.shape)
        print("log_prob shape", log_prob.shape)
        weighted_log_prob = log_prob * fest_values[i]
        weighted_log_prob = weighted_log_prob.reshape([contour_n, contour_n + 1])

        # Save the weighted log_prob
        log_prob_collection[i] = weighted_log_prob

    # Sum the weighted log_prob
    log_prob = torch.log(torch.mean(log_prob_collection.exp(), dim=0))
    return xs, ys, log_prob.detach().cpu().numpy()


def visualize_usage_subtracting_nn_means():
    # Load the data
    data_file = "../data/nodes/node_usages_halton_graph_with_obstacles_2.json"
    data = read_from_json(data_file)
    graph_file = data["graph_file"]
    g = get_graph_from_grahml(graph_file, "state")

    # Get the usages
    usages = [data["node_usage_counts"][str(node)] for node in g.nodes]
    usages_cp = deepcopy(usages)

    # Get the mean of neighbors
    for node in g.nodes:
        neighbors = g.neighbors(node)
        neighbors_usages = [data["node_usage_counts"][str(neighbor)] for neighbor in neighbors]
        mean = np.mean(neighbors_usages)
        usages[node] = usages[node] - mean

    # Visualize the usages
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    import networkx as nx

    # Draw the graph
    acc= nx.draw_networkx_nodes(g,
                           pos=nx.get_node_attributes(g, 'state'),
                           nodelist=g.nodes,
                           node_color=usages,
                           node_size=30,
                           cmap=plt.cm.bwr,
                           ax=ax[0])

    ac = nx.draw_networkx_nodes(g,
                           pos=nx.get_node_attributes(g, 'state'),
                           nodelist=g.nodes,
                           node_color=usages_cp,
                           node_size=30,
                           cmap=plt.cm.bwr,
                           ax=ax[1])

    ax[0].set_title("Usage - Mean of neighbors")
    ax[1].set_title("Usage")
    set_plot_square(ax[0])
    set_plot_square(ax[1])

    # Visualize the colorbar
    plt.colorbar(ac)


visualize_usage_subtracting_nn_means()
plt.show()