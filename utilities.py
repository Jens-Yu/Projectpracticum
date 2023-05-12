import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import copy
import torch


class PlanningRequest:
    def __init__(self, start, goal, dof=2) -> None:
        self.dof = dof
        self.obstacles = None
        self.planning_scene = None

        self.start = start
        self.goal = goal
        self.obstacles = None

    # def random_requst(self):
    #     start = self.range_low +  np.random.rand(self.dof) * (self.range_high - self.range_low)
    #     goal = self.range_low +  np.random.rand(self.dof) * (self.range_high - self.range_low)


class PlanningResult:
    def __init__(self) -> None:
        self.has_solution = False
        self.solution_path = None
        self.index_path = None
        self.checked_counts = 0
        self.checked_dict = {}


def form_cov_matrix(std: torch.Tensor, co_eff: torch.Tensor, k=1):
    cov_mat_list = []
    n_dim = int(torch.flatten(std).shape[0] / k)
    n_co_eff = int(torch.flatten(co_eff).shape[0] / k)

    assert (n_co_eff == n_dim*(n_dim-1)/2)
    std = std.reshape([-1, n_dim])

    for ik in range(k):
        eff = co_eff.squeeze()[n_co_eff*ik: n_co_eff*(ik+1)]
        cov_mat = torch.diag(torch.square(std[ik]))

        cnt = 0
        for row in range(1, n_dim):
            for col in range(row):
                cov_mat[row, col] = eff[cnt]*std[ik][row]*std[ik][col]
                cnt += 1

        cov_mat_list.append(cov_mat)
    return cov_mat_list


def generate_random_obstacles(max_number=10, min_number=3, max_radius=0.3, space_range=[-1, 1]):
    n_obstable = max(min_number, int(np.random.rand() * max_number))
    obstacles = []
    for i in range(n_obstable):
        has_new_obs = False
        while not has_new_obs:
            x = space_range[0] + np.random.rand()* (space_range[1] - space_range[0])
            y = space_range[0] + np.random.rand()* (space_range[1] - space_range[0])
            obs ={"x": x, "y": y}
            radius = np.random.rand()*max_radius
            dist_to_origin = obs["x"]**2 + obs["y"]**2 - radius
            has_new_obs = dist_to_origin > 0.0001
        
        obs.update({"r": (radius)})
        obstacles.append(obs)
    
    return obstacles


def generate_grid_like_graph(x_size=10, y_size=10):
    g = nx.Graph()
    range_min = -np.pi
    range_max = np.pi
    # x_list = np.linspace(range_min, range_max, x_size).tolist()
    # y_list = np.linspace(range_min, range_max, y_size).tolist()

    x_list = np.linspace(range_min, range_min*0.6, 2).tolist()
    x_list = x_list + np.linspace(range_min*0.4, range_max*0.4, 6).tolist()
    x_list = x_list + np.linspace(range_max*0.6, range_max, 2).tolist()

    y_list = np.linspace(range_min, range_min * 0.7, 2).tolist()
    y_list = y_list + np.linspace(range_min * 0.6, range_min * 0.5, 2).tolist()
    y_list = y_list + np.linspace(range_min * 0.2, range_max * 0.2, 2).tolist()
    y_list = y_list + np.linspace(range_max * 0.5, range_max * 0.6, 2).tolist()
    y_list = y_list + np.linspace(range_max * 0.7, range_max, 2).tolist()

    x_list, y_list = y_list, x_list
    xy_list = [np.array([x, y]) for x in x_list for y in y_list]
    x_step_diff = x_list[1] - x_list[0]
    y_step_diff = y_list[1] - y_list[0]

    for idx, sample in enumerate(xy_list):
        max_nn_dist = np.linalg.norm(np.array([x_step_diff, y_step_diff]))
        max_nn_number = 8

        def is_nn(nn):
            close_enough = np.linalg.norm(g.nodes[nn]["state"] - sample) <= max_nn_dist
            if not close_enough:
                return False

            return True

        nns = list(filter(is_nn, g.nodes))
        nns = sorted(nns,
                     key=lambda x: np.linalg.norm(g.nodes[x]["state"] - sample))
        if len(nns) > max_nn_number:
            nns = nns[:max_nn_number]

        g.add_nodes_from([(idx, {"state": sample})])
        for nn in nns:
            g.add_edge(nn, idx)
    return g


def make_numpy_graph_to_graphml_ok(graph:nx.Graph, attr:str):
    n = graph.number_of_nodes()
    for i in range(n):
        array_str = np.array2string(graph.nodes[i][attr], precision=3,
                    separator=',', suppress_small=True)
        graph.nodes[i][attr] = array_str
    return graph


def visualize_graphs(graphs:list, attr:str, show_edges=False):
    n_graphs = len(graphs)
    fig, axs = plt.subplots(
        nrows=n_graphs, ncols=1, sharex=False, sharey=False, figsize=(20, 14))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i in range(n_graphs):
        g = graphs[i]
        state = nx.get_node_attributes(g, attr)
        if show_edges:
            edge_colors = ["black"] * g.number_of_edges()
            nx.draw_networkx_edges(
                g, state, alpha=0.4, ax=axs[i], edge_color=edge_colors)

        nx.draw_networkx_nodes(
            g,
            state,
            nodelist=g.nodes,
            node_size=60,
            cmap=plt.cm.Reds_r,
            ax=axs[i]
        )
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].set_xlim(-3.5, 3.5)
        axs[i].set_ylim(-3.5, 3.5) # slightly greater than pi

    plt.show()


# TODO: I don't know what this is..
def visualize_graphs_compact(graphs:list, attr:str, show_edges=False):
    n_graphs = len(graphs)
    fig, axs = plt.subplots(
        nrows=1, ncols=1, sharex=False, sharey=False, figsize=(20, 14))

    if not isinstance(axs, np.ndarray):
        axs = [axs]
    
    color=["red", "blue"]

    for i in range(n_graphs):
        g = graphs[i]
        state = nx.get_node_attributes(g, attr)

        if show_edges:
            edge_colors = ["while"] * g.number_of_edges()
            nx.draw_networkx_edges(
                g, state, alpha=0.4, ax=axs[i], edge_color=edge_colors)

        nx.draw_networkx_nodes(
            g,
            state,
            nodelist=g.nodes,
            node_size=60,
            node_color=color[i],
            cmap=plt.cm.Reds_r,
            ax=axs[0]
        )

    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlim(-3.5, 3.5)
    axs[0].set_ylim(-3.5, 3.5)
    plt.show()


def visualize_tensor_graph(data, show_edges=False):
    """
    Plot the sub-graph given the Data
    :param data: torch_geometic.data.Data
    :param show_edges: bool
    :return: None
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))

    from torch_geometric.utils.convert import to_networkx
    len_data = data.x.shape[0]
    g = to_networkx(data, to_undirected=True)
    data_x = data.x.cpu().detach().numpy()

    if show_edges:
        edge_colors = ["black"] * g.number_of_edges()
        nx.draw_networkx_edges(
            g, data_x, alpha=0.4, ax=axs, edge_color=edge_colors)

    node_color = ["blue"] * len_data
    node_color[-1] = "red"

    nx.draw_networkx_nodes(
        g,
        data_x,
        nodelist=g.nodes,
        node_size=60,
        cmap=plt.cm.Reds_r,
        node_color=node_color,
        ax=axs
    )
    axs.set_aspect('equal', adjustable='box')
    axs.set_xlim(-3.5, 3.5)
    axs.set_ylim(-3.5, 3.5) # slightly greater than pi

    plt.show()


def get_graph_from_grahml(file_path:str, attr:str):
    G = nx.read_graphml(file_path, node_type=int)
    # jnt_corrdinates = np.array([], dtype=np.float64)
    
    for node in G.nodes:
        coodinates = G.nodes[node][attr]
        xyz = coodinates[1:-1].split(",")
        xyz = [float(string) for string in xyz]
        G.nodes[node][attr] = np.array(xyz)
    return G


def save_graph_to_file(g, graph_file: str):
    g = make_numpy_graph_to_graphml_ok(g, "state")
    # Save graph
    nx.write_graphml(g, graph_file)


def load_planning_requests(dataset_file):
    # Get planning dataset
    if dataset_file is not None:
        pre_gen_dataset = []
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
            for item in dataset.values():
                start = np.array(item["start"])
                goal = np.array(item["goal"])
                req = PlanningRequest(start, goal)
                req.obstacles = item["obstacles"]
                pre_gen_dataset.append(req)
        return pre_gen_dataset
    else:
        raise ValueError("File for dataset is not given!")


def read_from_json(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset


def write_to_json(data_dict, json_file):
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)


def write_planning_request_to_json(file_name, planning_requests: list):
    json_dict = {}
    for i, req in enumerate(planning_requests):
        planning_req = {"start": req.start.tolist(),
                        "goal": req.goal.tolist(),
                        "obstacles": req.obstacles}
        json_dict[i] = planning_req

    write_to_json(json_dict, file_name)


def write_planning_solutions_to_json(file_name, solutions: list):
    json_dict = {}
    for i, sol in enumerate(solutions):
        json_dict[i] = sol

    write_to_json(json_dict, file_name)


def write_dataset_to_json(basis_file_name, planning_requests: list, solutions: list):
    planning_request_file = "plreq_" + basis_file_name
    planning_solution_file = "plsol_" + basis_file_name
    write_planning_request_to_json(planning_request_file, planning_requests)


def add_node(g, node_state, idx=None, remove_identical=False):
    max_nn_dist = np.pi
    max_nn_number = 20

    if idx is None:
        idx = g.number_of_nodes()

    def is_nn(nn):
        close_enough = np.linalg.norm(g.nodes[nn]["state"] - node_state) < max_nn_dist
        return close_enough

    # TODO(Xi): Consider add collision checking function
        # if self.static_obstacles is not None:
        #     valid_edge, _ = self.check_validity(g.nodes[nn]["state"], node_state, self.static_obstacles)
        #     return valid_edge
        # return True

    nns = list(filter(is_nn, g.nodes))
    nns = sorted(nns,
        key=lambda x: np.linalg.norm(g.nodes[x]["state"] - node_state))

    if remove_identical:
        has_identical = True
        while has_identical:
            nn = nns[0]
            has_identical = np.linalg.norm(g.nodes[nn]["state"] - node_state) < 1e-6
            if has_identical:
                nns = nns[1:]
                g.remove_node(nn)

    if len(nns) > max_nn_number:
        nns = nns[:max_nn_number]

    g.add_nodes_from([(idx, {"state": node_state})])
    for nn in nns:
        g.add_edge(nn, idx)


def process_results_list(result_list):
    result_dict = {}
    for i, pr in enumerate(result_list):
        sol_path = [a.tolist() for a in pr.solution_path]
        pr_dict = {"has_solution": pr.has_solution,
                   "solution_path": sol_path,
                   "index_path": pr.index_path,
                   "checked_counts": pr.checked_counts,
                   "checked_dict": pr.checked_dict}
        result_dict.update({i: pr_dict})
    return result_dict


def get_result_statistics(result_dict):
    success_sum = 0
    checked_counts_sum = 0
    path_length_sum = 0
    path_list = []
    index_path_list = []
    visited_nodes = []
    blocked_nodes = {}
    for result in result_dict.values():
        success_sum += 1 if result["has_solution"] else 0
        checked_counts_sum += result["checked_counts"]

        new_visited_nodes = list(result["checked_dict"].keys())
        new_visited_nodes = list(filter(lambda x: x not in visited_nodes, new_visited_nodes))
        visited_nodes = visited_nodes + new_visited_nodes

        for n, n_is_valid in result["checked_dict"].items():
            blocked_time = blocked_nodes.get(n, 0)
            if not n_is_valid:
                blocked_nodes.update({n: blocked_time+1})

        if result["has_solution"]:
            s_path = [np.array(s) for s in result["solution_path"]]
            diffs = [s_path[i+1] - s_path[i] for i in range(len(s_path) - 1)]
            path_length_sum += sum([np.linalg.norm(diff) for diff in diffs])
            path_list.append(result["solution_path"])
            index_path_list.append(result["index_path"])
        else:
            path_list.append([])

    return {"success_sum": success_sum,
            "checked_counts": checked_counts_sum,
            "path_length_sum": path_length_sum,
            "path_list": path_list,
            "index_path_list": index_path_list,
            "visited_nodes": visited_nodes,
            "blocked_nodes": blocked_nodes}


def visualize_graph_ground_truth(file_name):
    json_dict = read_from_json(file_name)
    graph_file = json_dict["0"]["graph"]
    g = get_graph_from_grahml(graph_file, "state")
    value_list = list(json_dict.values())
    characters = [v["character"] for v in value_list]
    print(characters)
    n_nodes = g.number_of_nodes()
    color_list = ["red" if characters[i] == "useless" else "blue" for i in range(n_nodes)]

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))

    state = nx.get_node_attributes(g, "state")
    edge_colors = ["black"] * g.number_of_edges()
    nx.draw_networkx_edges(
        g, state, alpha=0.4, ax=axs, edge_color=edge_colors)

    nx.draw_networkx_nodes(
            g,
            state,
            nodelist=g.nodes,
            node_size=60,
            ax=axs,
            node_color=color_list
        )
    axs.set_aspect('equal', adjustable='box')
    axs.set_xlim(-3.5, 3.5)
    axs.set_ylim(-3.5, 3.5)  # slightly greater than pi

    plt.show()


def visualize_distributions(root):
    number_of_graph = 1
    # TODO(Xi): Just two examples for good and useless nodes
    good_node_paths = [os.path.join(
            root, "nodes/good_nodes_{}.json".format(i)) for i in range(number_of_graph)]
    useless_node_paths = [os.path.join(
            root, "nodes/useless_nodes_{}.json".format(i)) for i in range(number_of_graph)]
    raw_paths = good_node_paths + useless_node_paths
    samples = []
    labels = []
    for raw_path in raw_paths:
        # Read data from `raw_path`.
        # Read important nodes or useless nodes
        json_dataset = read_from_json(raw_path)
        for json_data in json_dataset.values():
            # TODO(Xi): Just some dummy conditions to test the dataset
            #gf = json_data["graph"]
            #g = get_graph_from_grahml(gf, "state")

            # good or useless
            character = json_data["character"]
            samples = samples + json_data["nodes"]
            labels = labels + [character]*len(json_data["nodes"])

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))
    sample_x = [sample[0] for sample in samples]
    sample_y = [sample[1] for sample in samples]
    color_nodes = ["red" if label == "useless" else "blue" for label in labels]

    ax.scatter(sample_x, sample_y, s=100, color=color_nodes)
    plt.show()


def visualize_dataset_distributions(data_loader):
    samples = []
    labels = []

    for data in data_loader:
        samples.append(data.x[data.test_mask].cpu().detach().numpy())
        labels.append(data.y[data.test_mask].cpu().detach().numpy())

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))
    samples = [sample.squeeze().tolist() for sample in samples]
    sample_x = [sample[0] for sample in samples]
    sample_y = [sample[1] for sample in samples]
    color_nodes = ["red" if label == 0 else "blue" for label in labels]

    ax.scatter(sample_x, sample_y, s=100, color=color_nodes)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)  # slightly greater than pi
    plt.show()


def visualize_global_dataset_distributions(data_loader):
    samples = []
    labels = []

    for data in data_loader:
        x = data.x.cpu().detach().numpy().tolist()
        y = data.y.cpu().detach().numpy().tolist()
        samples = samples + x
        labels = labels + y

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))
    sample_x = [sample[0]*np.pi for sample in samples]
    sample_y = [sample[1]*np.pi for sample in samples]
    color_nodes = ["red" if label == 0 else "blue" for label in labels]

    ax.scatter(sample_x, sample_y, s=100, color=color_nodes)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)  # slightly greater than pi
    plt.show()


def visualize_cost_data_file(data_file, original_path=None):
    # Get graph
    data = read_from_json(data_file)
    data_idx = list(data.keys())
    n_data = len(data_idx)
    graph_file = data[data_idx[0]]["graph"]
    g = get_graph_from_grahml(graph_file, "state")

    # Check graph
    # assert (g.number_of_nodes() == n_data)
    # for i in g.nodes:
    #     diff = g.nodes[i] - np.array(data[str(i)])
    #     assert (np.linalg.norm(diff) < 1e-6)

    # Get data
    node_states = [g.nodes[i]["state"] for i in range(g.number_of_nodes)]
    path_length_sum_diff = [data[idx].get("path_length_sum_diff", 0) for idx in data_idx]
    success_sum_diff = [data[idx].get("success_sum_diff", 0) for idx in data_idx]
    success_sum_before = [data[idx].get("success_sum_before", 0) for idx in data_idx]
    checked_counts_diff = [data[idx].get("checked_counts_diff", 0) for idx in data_idx]
    checked_counts_before = [data[idx].get("checked_counts_before", 0) for idx in data_idx]
    previous_ground_truth = [0 if success_sum_before[idx] == 0 or success_sum_diff[idx] == 0
                             else 1 for idx in range(n_data)]
    original_node_usages_list = []

    if original_path is not None:
        original_path_data = read_from_json(original_path)
        original_node_usages = original_path_data["node_usages_without_obstacles"]
        original_node_usages_list = list(original_node_usages.values())
        max_v = max(original_node_usages_list)
        original_node_usages_list = [v/max_v for v in original_node_usages_list]
    #
    data_list = [path_length_sum_diff,
                 success_sum_diff,
                 success_sum_before,
                 checked_counts_diff,
                 previous_ground_truth,
                 original_node_usages_list]

    # TODO: need to conver the data_list to color_list
    title_list = ["path_length_sum_diff",
                  "success_sum_diff",
                  "success_sum_before",
                  "checked_counts_diff",
                  "previous_ground_truth",
                  "success_sum_without_obstacles"]

    # Visualize data
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 30))
    nc_list = []

    plot_number = 5 if original_path is None else 6
    for i in range(plot_number):
        ax = axs[i//3][i % 3]
        # edge_colors = ["black"] * g.number_of_edges()
        # nx.draw_networkx_edges(
        #     g, node_states, alpha=0.4, ax=axs[i], edge_color=edge_colors)

        nc = nx.draw_networkx_nodes(
            g,
            node_states,
            nodelist=g.nodes,
            node_size=60,
            cmap=plt.cm.bwr_r,
            node_color=data_list[i],
            ax=ax
        )
        nc_list.append(nc)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)  # slightly greater than pi
        ax.set_title(title_list[i])

    annot = axs[1][2].annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),)
    annot.set_visible(False)

    def hover(event):
        vis = annot.get_visible()
        for ncc in nc_list:
            cont, ind = ncc.contains(event)
            if cont:
                idx = ind["ind"][0]
                text = f"idx: {idx} \n" \
                       f"node: {node_states[idx]} \n" \
                       f"path_length_sum_diff: {path_length_sum_diff[idx]} \n" \
                       f"success_sum_diff: {success_sum_diff[idx]} \n" \
                       f"success_sum_before: {success_sum_before[idx]} \n" \
                       f"checked_counts_diff: {checked_counts_diff[idx]} \n" \
                       f"checked_counts_before: {checked_counts_before[idx]} \n" \
                       f"previous_ground_truth: {previous_ground_truth[idx]} \n"

                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                break
            elif vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def visualize_plan_b_data(proccessed_data_file):
    # Load data
    data = read_from_json(proccessed_data_file)
    graph_file = "../data/graphs/structure_graph.graphml"
    g = get_graph_from_grahml(graph_file, "state")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(14, 14))

    node_states = nx.get_node_attributes(g, "state")

    edge_colors = ["black"] * g.number_of_edges()
    nx.draw_networkx_edges(g, node_states, alpha=0.4, ax=axs, edge_color=edge_colors)

    nc = nx.draw_networkx_nodes(
        g,
        node_states,
        nodelist=g.nodes,
        node_size=60,
        cmap=plt.cm.bwr_r,
        node_color=np.zeros(g.number_of_nodes()),
        ax=axs
    )

    axs.set_aspect('equal', adjustable='box')
    axs.set_xlim(-3.5, 3.5)
    axs.set_ylim(-3.5, 3.5)  # slightly greater than pi

    annot = axs.annotate("", xy=(0, 0), xytext=(0, 400), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w"), )
    annot.set_visible(False)

    def hover(event):
        vis = annot.get_visible()
        cont, ind = nc.contains(event)
        if cont:
            idx = ind["ind"][0]
            plan_b_data = data[str(idx)]["planB_nodes"]
            text = f"idx: {idx} \n" \
                   f"node: {node_states[idx]} \n" \
                   f"number of plan_b: {len(plan_b_data)}"

            node_color = np.zeros(g.number_of_nodes())
            for i in plan_b_data:
                node_color[i] += 1
            normalize_factor = max(np.max(node_color), 1)
            node_color = node_color / normalize_factor
            node_alpha = node_color
            cm = plt.cm.bwr_r
            node_color = [cm(c) for c in node_color]

            nc.set_alpha(node_alpha)
            nc.set(color=node_color, facecolor=node_color)
            annot.set_text(text)
            annot.set_visible(True)

            fig.canvas.draw_idle()

        elif vis:
            node_color = np.zeros(g.number_of_nodes())
            cm = plt.cm.bwr_r
            node_color = [cm(c) for c in node_color]

            nc.set_alpha(1)
            nc.set(color=node_color, facecolor=node_color)
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def save_dataset_visualizations(dataset_names, title=None):
    """
    Generate visualizations for the nodes colored with labels
    Three views of the visualization: combined view, "important" labeled view, "useless" labeled view 
    """
    pass


def filter_out_local_graph(global_graph: nx.Graph, node_idx, hop, remove_close_nodes=False, threshold=0.15):
    local_graph = nx.ego_graph(global_graph, node_idx, radius=hop)
    # result from https://stackoverflow.com/questions/18393842/k-th-order-neighbors-in-graph-python-networkx
    return local_graph


def filter_out_close_nodes(g, labels, threshold=0.15):
    edges = copy.deepcopy(g.edges)
    removed_list = []
    for u, v in edges:
        if u in removed_list or v in removed_list:
            continue

        u_state = g.nodes[u]["state"]
        v_state = g.nodes[v]["state"]
        if np.linalg.norm(u_state - v_state) < threshold:
            if labels[u] == labels[v]:
                g.remove_node(u)
                removed_list.append(u)
            elif labels[u] == [0, 1]:  # Label useless
                g.remove_node(u)
                removed_list.append(u)
            else:
                g.remove_node(v)
                removed_list.append(v)
            # print(f"Removed! N nodes {g.number_of_nodes()}")


def fix_file_path(file_path):
    return ".." + file_path.split("..")[-1]


def set_plot_square(ax, x_lim=(-3.5, 3.5), y_lim=(-3.5, 3.5)):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])  # slightly greater than pi


