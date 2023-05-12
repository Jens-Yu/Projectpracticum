import torch.nn.functional as F
from pyg_dataset import *
from torch_geometric.loader import DataLoader
from model import GCNGMM, GMM, FEST
import os

import torch.distributions as D
from torch.autograd import Variable
from train_GMM import random_data, prepare_data, form_cov_matrix

from planning import get_2dof_planning_environment
from generate_classification_dataset import filter_out_important_samples
import networkx as nx


def test(config, k_gmm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    graph_file = config["graph_file"]
    model_file = config["model_file"]

    g = get_graph_from_grahml(graph_file, "state")
    n_nodes = g.number_of_nodes()
    node_states = nx.get_node_attributes(g, "state")
    node_states = [torch.from_numpy(node_states[i]) for i in range(n_nodes)]
    node_states_tensor = torch.stack(node_states).to(device)
    node_states_np = node_states_tensor.detach().cpu().numpy()

    node_states_x = [ns[0] for ns in node_states]
    node_states_y = [ns[1] for ns in node_states]

    # Get random data for testing
    d = prepare_data(config, "test")
    x, y_gt = random_data(d)

    # Get the ground truth data and also the node that is blocked
    node_states_x_gt = [ns[0] for ns in y_gt] + [x[0][0]]
    node_states_y_gt = [ns[1] for ns in y_gt] + [x[0][1]]

    # Initializing the model
    model = GMM(k_gmm).double()
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    # Prepare the grids for contour
    contour_n = 80
    xs = np.linspace(-np.pi, np.pi, contour_n)
    ys = np.linspace(-np.pi, np.pi, contour_n+1)

    nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    nodes_tensor = torch.stack(nodes_tensor).to(device)

    n_cols = 3
    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(8, 8))
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(x).to(device)
        mean, log_std, co_eff, log_w = model(x)
        mean = mean.reshape(k_gmm, -1)
        sigma = log_std.reshape(k_gmm, -1).exp()
        # Construct the covariance matrix
        cov_mat_list = form_cov_matrix(sigma, co_eff, k_gmm)

        dist_list = [D.MultivariateNormal(mean[ik], scale_tril=cov_mat_list[ik])
                     for ik in range(k_gmm)]

        # Compute log probabilities
        log_w = log_w.squeeze()
        exp_log_py_list = [torch.exp(dist_list[id].log_prob(nodes_tensor) + log_w[id])
                           for id in range(k_gmm)]

        exp_log_py = torch.stack(exp_log_py_list)
        exp_log_py = torch.sum(exp_log_py, dim=0)

        pr = exp_log_py.reshape([contour_n, contour_n+1]).detach().cpu().numpy()
        axs[0].contourf(xs, ys, np.transpose(pr), vmin=0, vmax=1)

        exp_log_py_list_2 = [torch.exp(dist_list[id].log_prob(node_states_tensor) + log_w[id])
                           for id in range(k_gmm)]

        exp_log_py_2 = torch.stack(exp_log_py_list_2)
        exp_log_py_2 = torch.sum(exp_log_py_2, dim=0)

        pr2 = exp_log_py_2.detach().cpu().numpy()
        axs[1].scatter(node_states_x, node_states_y, c=pr2, cmap=plt.cm.bwr_r)
        nx.draw_networkx_edges(
            g, node_states_np, alpha=0.1, ax=axs[1])

    color = ["blue"] * len(node_states_x_gt)
    color[-1] = "red"
    axs[2].scatter(node_states_x_gt, node_states_y_gt, c=color, alpha=0.1)

    for i_sp in range(n_cols):
        axs[i_sp].set_aspect('equal', adjustable='box')
        axs[i_sp].set_xlim(-3.5, 3.5)
        axs[i_sp].set_ylim(-3.5, 3.5)  # slightly greater than pi

    plt.show()
    #https: // gist.github.com / lirnli / a10629fd068384d114c301db18e729c8


def shift_using_gmm(gmm_model_path, weighting_model_path, k_gmm, node_states):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    if isinstance(node_states, list):
        node_states_tensor = torch.stack(node_states).to(device)
    else:
        node_states_tensor = torch.from_numpy(node_states).to(device)
    n_nodes = node_states_tensor.shape[0]

    # Load GMM model
    model = GMM(k_gmm)
    model.load_state_dict(torch.load(gmm_model_path))
    model.double()
    model = model.to(device)

    # Load weighting model
    model_w = FEST().to(device)
    model_w.load_state_dict(torch.load(weighting_model_path))
    model_w.double()

    # Ensure the pytorch return gradients with respect to the gradients
    mean, log_std, co_eff, log_w = model(node_states_tensor)

    # Mean shape: [n_data, k_gmm*2]
    mean = mean.reshape([n_nodes, k_gmm, -1])
    sigma = log_std.reshape([n_nodes, k_gmm, -1]).exp()
    co_eff = co_eff.reshape([n_nodes, k_gmm, -1])

    # gdt_k = torch.zeros(n_nodes, n_nodes, 2).to(device)

    # compute the weight
    mixture_list = []
    variable_list = []
    for i in range(n_nodes):
        cov_mat_list = form_cov_matrix(sigma[i], co_eff[i], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[i], scale_tril=cov_mat_tensor), 0)  # Not sure about 1
        w = D.Categorical(log_w[i].squeeze().exp())

        # Mixture model given the node i
        mixture = D.MixtureSameFamily(w, dist)
        mixture_list.append(mixture)
        variable_list.append(Variable(node_states_tensor[i], requires_grad=True))

    for j in range(n_nodes):
        gdt = torch.zeros(n_nodes, 2).to(device)
        node_states_tensor_v = variable_list[j]
        prob_w = model_w(node_states_tensor)
        for i in range(n_nodes):
            # We should try to vertorize the computation
            if i == j:
                continue

            # To all nodes: Plan B log_py for node i
            log_py = mixture_list[i].log_prob(node_states_tensor_v)
            log_py_w = log_py*prob_w[i]
            gdt[i, :] = torch.autograd.grad(log_py_w, node_states_tensor_v, retain_graph=True)[0].data * 0.15

        bw = 1.5
        kernel_product = torch.exp(-0.5/bw*torch.norm(node_states_tensor_v - node_states_tensor, dim=1)*prob_w)
        a = torch.autograd.grad(kernel_product.mean(), node_states_tensor_v, retain_graph=True)[0].data

        node_states_tensor[j, :] += torch.mean(gdt, dim=0) - a*5
        mean_v, log_std_v, co_eff_v, log_w_v = model(node_states_tensor[j, :].reshape([-1, 2]))
        mean[j, :, :] = mean_v.reshape([k_gmm, -1])
        sigma[j, :, :] = log_std_v.reshape([k_gmm, -1]).exp()
        co_eff[j, :, :] = co_eff_v.reshape([k_gmm, -1])
        log_w[j, :] = log_w_v.reshape([k_gmm])

        cov_mat_list = form_cov_matrix(sigma[j], co_eff[j], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[j], scale_tril=cov_mat_tensor), 0)  # Not sure about 1
        w = D.Categorical(log_w[j].squeeze().exp())

        # Mixture model given the node i
        mixture_list[j] = D.MixtureSameFamily(w, dist)

    node_states_modified = node_states_tensor.detach().cpu().numpy()

    return node_states_modified


def plan_b_importance(model_path, k_gmm, node_states, est_nn):
    """
    Given the GMM model, compute the importance of all regions.
    This function may not make much sense.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    if isinstance(node_states, list):
        node_states_tensor = torch.stack(node_states).to(device)
    else:
        node_states_tensor = torch.from_numpy(node_states).to(device)

    n_nodes = node_states_tensor.shape[0]
    # Load GMM model
    model = GMM(k_gmm)
    model.load_state_dict(torch.load(model_path))
    model.double()
    model = model.to(device)

    # Ensure the pytorch return gradients with respect to the gradients
    mean, log_std, co_eff, log_w = model(node_states_tensor)

    # Mean shape: [n_data, k_gmm*2]
    mean = mean.reshape([n_nodes, k_gmm, -1])
    sigma = log_std.reshape([n_nodes, k_gmm, -1]).exp()
    co_eff = co_eff.reshape([n_nodes, k_gmm, -1])

    contour_n = 80
    xs = np.linspace(-np.pi, np.pi, contour_n)
    ys = np.linspace(-np.pi, np.pi, contour_n + 1)

    nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    nodes_tensor_ct = torch.stack(nodes_tensor).to(device)
    w_prob = torch.zeros(contour_n*(contour_n + 1)).to(device)

    for i in range(n_nodes):
        # We should try to vertorize the computation

        cov_mat_list = form_cov_matrix(sigma[i], co_eff[i], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[i], scale_tril=cov_mat_tensor), 0)  # Not sure about 1
        w = D.Categorical(log_w[i].squeeze().exp())

        # Mixture model given the node i
        mixture = D.MixtureSameFamily(w, dist)

        # Plan B log_py for all nodes
        log_py = mixture.log_prob(nodes_tensor_ct)

        # Weight of node i
        nn_w = est_nn(nodes_tensor_ct[i])

        log_py = log_py*nn_w

        w_prob += log_py

    w_prob = w_prob.reshape([contour_n, contour_n + 1]).detach().cpu().numpy()
    return w_prob


def visualize_region_values_learned_from_path(model_w_file, model_wo_file):
    """
    Load a GMM model.
    Train value estimation models for the data with and without obstacles.
    Use the trained model for data with obstacles and combine it with GMM ->
        Visualize the output of GMM given the nodes -> Where are important for these nodes
    (Did not consider the existence of other nodes)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    model_w = FEST().double()
    model_wo = FEST().double()

    model_w.load_state_dict(torch.load(model_w_file))
    #model_wo.load_state_dict(torch.load(model_wo_file))

    contour_n = 80
    xs = np.linspace(-np.pi, np.pi, contour_n)
    ys = np.linspace(-np.pi, np.pi, contour_n + 1)

    nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    nodes_tensor = torch.stack(nodes_tensor).to(device)

    est_wo = model_wo(nodes_tensor)
    est_w = model_w(nodes_tensor)

    est_wo_np = est_wo.reshape([contour_n, contour_n + 1]).detach().cpu().numpy()
    est_w_np = est_w.reshape([contour_n, contour_n + 1]).detach().cpu().numpy()

    n_cols = 2
    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(14, 8))
    axs[0].contourf(xs, ys, np.transpose(est_wo_np))
    axs[1].contourf(xs, ys, np.transpose(est_w_np))

    title_list = ["Value estimation without obstacles",
                  "Value estimation with obstacles"]
    for i_sp in range(n_cols):
        axs[i_sp].set_aspect('equal', adjustable='box')
        axs[i_sp].set_xlim(-3.5, 3.5)
        axs[i_sp].set_ylim(-3.5, 3.5)  # slightly greater than pi
        axs[i_sp].set_title(title_list[i])


def sample_from_gmm(model_path, k_gmm, node_states):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    if isinstance(node_states, list):
        node_states_tensor = torch.stack(node_states).to(device)
    elif isinstance(node_states, np.ndarray):
        node_states_tensor = torch.from_numpy(node_states).to(device)
    else:
        node_states_tensor = node_states
    n_nodes = node_states_tensor.shape[0]

    # Load GMM model
    model = GMM(k_gmm)
    model.load_state_dict(torch.load(model_path))
    model.double()
    model = model.to(device)

    # Ensure the pytorch return gradients with respect to the gradients
    mean, log_std, co_eff, log_w = model(node_states_tensor)

    # Mean shape: [n_data, k_gmm*2]
    mean = mean.reshape([n_nodes, k_gmm, -1])
    sigma = log_std.reshape([n_nodes, k_gmm, -1]).exp()
    co_eff = co_eff.reshape([n_nodes, k_gmm, -1])

    new_samples = torch.zeros(n_nodes, 2)
    for i in range(n_nodes):
        # We should try to vertorize the computation
        cov_mat_list = form_cov_matrix(sigma[i], co_eff[i], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[i], scale_tril=cov_mat_tensor), 0) # Not sure about 1
        w = D.Categorical(log_w[i].squeeze().exp())
        mixture = D.MixtureSameFamily(w, dist)
        new_samples[i, :] = mixture.sample()

    node_states_tensor = torch.concat([node_states_tensor, new_samples], dim=0)
    return node_states_tensor


def visualize_hard_planning_request_result(config, save_figure=False):
    """
    Visualize the hard planning request,
    blocked nodes in the graph and their plan-B regions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    g_file = config["graph"]
    requests_file = config["requests"]
    planning_results_file = config["original_path"]
    request_idx = config["request_idx"]
    g = get_graph_from_grahml(g_file, "state")
    planning_requests = load_planning_requests(requests_file)

    hard_req = planning_requests[request_idx]
    planning_results = read_from_json(planning_results_file)["index_solution_paths"]
    result_hard_req = planning_results[str(request_idx)]

    # Examine if the result matches the request
    start = result_hard_req["start"]
    print("hard_req.start: ", hard_req.start)
    print("start: ", start)

    # Get blocked nodes TODO: Consider extend this to all blocked nodes
    checked_list = result_hard_req["with_obstacles_checked_nodes"]
    solution_without_obs = result_hard_req["without_obstacles"]
    solution_with_obs = result_hard_req["with_obstacles"]

    # Get planning agent to compute the exact blocked points
    pl_env = get_2dof_planning_environment()
    pl_env.G = copy.deepcopy(g)
    blocked_list = []
    solution_without_obs = [hard_req.start] + \
                           [g.nodes[wp_idx]["state"] for wp_idx in solution_without_obs] +\
                           [hard_req.goal]

    for id_n in range(len(solution_without_obs)-1):
        node = solution_without_obs[id_n]
        next_node = solution_without_obs[id_n+1]
        is_valid, n_checking = pl_env.check_validity(
            node, next_node, hard_req.obstacles)
        if not is_valid:
            if n_checking == 1:
                blocked_list.append(next_node)
            else:
                diff = np.linalg.norm(next_node - node)
                normalized_vector = (next_node - node)/diff
                blocked_node = (n_checking-2)*pl_env.resolution*normalized_vector + node
                blocked_list.append(blocked_node)

    blocked_nodes_tensor = [torch.from_numpy(ns).to(device) for ns in blocked_list]
    blocked_nodes_tensor = torch.stack(blocked_nodes_tensor).to(device)
    n_blocked_nodes = len(blocked_list)

    # Load models
    k_gmm = 20
    gmm_model_file = config["model_file"]
    weighting_model_file = "../data/models/weighting_model.pt"

    model_gmm = GMM(k_gmm).double().to(device)
    model_w = FEST().double().to(device)

    model_gmm.load_state_dict(torch.load(gmm_model_file))
    # model_w.load_state_dict(torch.load(weighting_model_file))

    # Create data points for the contour
    contour_n = 80
    xs = np.linspace(-np.pi, np.pi, contour_n)
    ys = np.linspace(-np.pi, np.pi, contour_n + 1)

    nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    nodes_tensor = torch.stack(nodes_tensor).to(device)

    # Compute the parameters of GMMs for all blocked nodes
    mean, log_std, co_eff, log_w = model_gmm(blocked_nodes_tensor)
    # Mean shape: [n_data, k_gmm*2]
    mean = mean.reshape([n_blocked_nodes, k_gmm, -1])
    sigma = log_std.reshape([n_blocked_nodes, k_gmm, -1]).exp()
    co_eff = co_eff.reshape([n_blocked_nodes, k_gmm, -1])

    # Compute the probability of all regions regarding the blocked nodes
    log_prob_cum = torch.zeros(contour_n*(contour_n+1)).to(device)
    for i_m in range(n_blocked_nodes):
        cov_mat_list = form_cov_matrix(sigma[i_m], co_eff[i_m], k_gmm)
        cov_mat_tensor = torch.stack(cov_mat_list)
        dist = D.Independent(D.MultivariateNormal(mean[i_m], scale_tril=cov_mat_tensor), 0)  # Not sure about 1
        w = D.Categorical(log_w[i_m].squeeze().exp())

        # Mixture model given the node i
        mixture = D.MixtureSameFamily(w, dist)

        # Get probability
        a = mixture.log_prob(nodes_tensor).exp()
        log_prob_cum += mixture.log_prob(nodes_tensor).exp()

    # Normalize the probability and make it log probability
    log_prob_cum = torch.log(log_prob_cum/n_blocked_nodes)
    log_prob_cum = log_prob_cum.reshape([contour_n, contour_n + 1]).detach().cpu().numpy()

    # Compute the estimated value of overall scores for each point in the configuration space
    est_w = model_w(nodes_tensor)
    est_w_np = est_w.reshape([contour_n, contour_n + 1]).detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 8))
    title_list = ["Estimated value with obstacles",
                  "Estimated log probability via GMM",
                  "RRT Connect solution (green=important)",
                  "Blocked node (cyan= start & goal)"]

    # Visualize estimated value with obstacles
    axs[0].contourf(xs, ys, np.transpose(est_w_np))

    # Visualize estimated of the log probability via GMM
    axs[1].contourf(xs, ys, np.transpose(log_prob_cum))

    # Visualize RRT result
    rrt_solution_with_obstacles = result_hard_req["rrt_solution_with_obstacles"]
    solution_x = [ns[0] for ns in rrt_solution_with_obstacles]
    solution_y = [ns[1] for ns in rrt_solution_with_obstacles]
    # Plot it with log probability via GMM
    axs[1].scatter(solution_x, solution_y, facecolor="blue")

    # Plot it alone
    important_samples = filter_out_important_samples(
        pl_env, hard_req, rrt_solution_with_obstacles)
    obstacle_space = pl_env.get_obstacle_space(hard_req)

    # Visualize the obstacles space
    obstacle_space_x = [ns[0] for ns in obstacle_space]
    obstacle_space_y = [ns[1] for ns in obstacle_space]
    axs[2].contourf(xs, ys, np.transpose(log_prob_cum))
    axs[2].scatter(obstacle_space_x, obstacle_space_y, facecolor="red", s=0.1)

    # The path planned by RRT
    axs[2].plot(solution_x, solution_y, color="blue")
    solution_x = [ns[0] for ns in important_samples]
    solution_y = [ns[1] for ns in important_samples]
    axs[2].scatter(solution_x, solution_y, facecolor="green")

    # The start and goal position
    goal = hard_req.goal
    req_x = [start[0], goal[0]]
    req_y = [start[1], goal[1]]
    axs[2].scatter(req_x, req_y, facecolor="cyan")

    # Visualize the graph
    node_states = [g.nodes[i]["state"] for i in g.nodes]
    nx.draw_networkx_nodes(
        g,
        node_states,
        nodelist=g.nodes,
        node_size=10,
        ax=axs[3]
    )
    nx.draw_networkx_edges(
        g, node_states, alpha=0.4, ax=axs[3])

    # Visualize the blocked nodes
    blocked_x = [ns[0] for ns in blocked_list]
    blocked_y = [ns[1] for ns in blocked_list]
    axs[3].scatter(obstacle_space_x, obstacle_space_y, facecolor="gray", s=0.1)
    axs[3].scatter(blocked_x, blocked_y, facecolor="red")
    axs[3].scatter(req_x, req_y, facecolor="cyan")
    axs[3].scatter(solution_x, solution_y, facecolor="green")

    # Visualize the path in the graph if there is one
    solution_with_obs = result_hard_req["with_obstacles"]
    if solution_with_obs:
        solution_x = [g.nodes[ns]["state"][0] for ns in solution_with_obs]
        solution_y = [g.nodes[ns]["state"][1] for ns in solution_with_obs]
        axs[3].scatter(solution_x, solution_y, facecolor="green")

    for i_a in range(4):
        axs[i_a].set_aspect('equal', adjustable='box')
        axs[i_a].set_xlim(-3.5, 3.5)
        axs[i_a].set_ylim(-3.5, 3.5)  # slightly greater than pi
        axs[i_a].set_title(title_list[i_a])

    if save_figure:
        fpath = f"../data/figure/{request_idx}.svg"
        plt.savefig(fpath, backend="svg")


def save_hard_request_figures():
    requests_file = "../data/pl_req/hard_pl_req_250_nodes.json"
    planning_requests = load_planning_requests(requests_file)
    n_req = len(planning_requests)
    for id_req in range(n_req):
        visualize_hard_planning_request_result(id_req, True)


if __name__ == "__main__":
    model_name = "GMM_halton_graph_with_obstacles_2_183.1765899658203.pt"
    path = os.path.join(os.getcwd(), "../data/models/{}".format(model_name))
    config = {"graph": "../data/graphs/halton_graph_with_obstacles_2.graphml",
              "plan_b_data": "../data/compare/processed_path_halton_graph_with_obstacles_2.json",
              "model_file": "../data/models/GMM_halton_graph_with_obstacles_2_183.1765899658203.pt",
              "original_path": "../data/compare/original_path_halton_graph_with_obstacles_2.json",
              "request_idx": 967,
              "requests": "../data/pl_req/easy_pl_req_250_nodes.json"}

    fest_model = "../data/models/FEST_value_model.pt"
    fest_model_wo = "../data/models/FEST_value_model_wo.pt"

    # Visualize the estimated value learn from the path data
    if os.path.isfile(fest_model) and os.path.isfile(fest_model_wo):
        visualize_region_values_learned_from_path(fest_model, fest_model_wo)

    # Visualize the hard planning request, blocked nodes in the graph and their plan-B regions
    visualize_hard_planning_request_result(config, False)
    plt.show()

    k_gmm = 20
    test(config, 20)

    weighting_model_name = "weighting_model.pt"
    w_path = os.path.join(os.getcwd(), "../data/models/{}".format(weighting_model_name))


    graph_file = "../data/graphs/structure_graph.graphml"
    g = get_graph_from_grahml(graph_file, "state")

    n_nodes = g.number_of_nodes()
    node_states = nx.get_node_attributes(g, "state")
    node_states = [torch.from_numpy(node_states[i]) for i in range(n_nodes)]
    node_states_x = [ns[0] for ns in node_states]
    node_states_y = [ns[1] for ns in node_states]
    node_states_ref = torch.stack(node_states)

    n_shift = 3
    fig, axs = plt.subplots(nrows=1, ncols=n_shift+1, figsize=(14, 8))
    axs[0].scatter(node_states_x, node_states_y, alpha=0.1)
    for i in range(n_shift):
        # Output is numpy
        node_states = shift_using_gmm(path, w_path, k_gmm, node_states)

        node_states_x_m = [ns[0] for ns in node_states]
        node_states_y_m = [ns[1] for ns in node_states]

        axs[i+1].scatter(node_states_x_m, node_states_y_m, alpha=0.1)
        axs[i+1].set_aspect('equal', adjustable='box')
        axs[i+1].set_xlim(-5, 5)
        axs[i+1].set_ylim(-5, 5)  # slightly greater than pi

    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlim(-5, 5)
    axs[0].set_ylim(-5, 5)  # slightly greater than pi
    plt.show()

    # # Save graph
    # graph_file = "../data/graphs/structure_graph_modified.graphml"
    # for i in range(n_nodes):
    #     g.nodes[i]["state"] = node_states[i]
    #
    # save_graph_to_file(g, graph_file=graph_file)



