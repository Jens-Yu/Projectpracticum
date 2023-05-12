# This file is used to visualize and print the results for classification.
# Currently, only the method benchmarking_two_roadmaps is used.

from pyg_dataset import *
from torch_geometric.loader import DataLoader
from model import GCN, GCNwLabels
from torch_geometric.data import Data
from copy import deepcopy

from planning import Planning
from manipulator import *

from joblib import Parallel, delayed


def print_results(file_names):
    if isinstance(file_names, str):
        file_names = [file_names]

    for file_name in file_names:
        json_dict = read_from_json(file_name)
        result_before = json_dict["before"]
        result_after = json_dict["after"]

        processed_result_before = get_result_statistics(result_before)
        processed_result_after = get_result_statistics(result_after)

        print("------------ Before ---------------")
        print(processed_result_before)
        print("------------ After  ---------------")
        print(processed_result_after)


def visualize_global_graph_classification(model_file, visual_type="result"):
    # Load data
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "../data")
    graph_file = cwd + "/../data/graphs/graph_250_nodes_99.graphml"

    # Define the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNwLabels()
    model_path = os.path.join(cwd, model_file)
    model.load_state_dict(torch.load(model_path))

    model.double()
    model = model.to(device)

    model.eval()
    y_pred_list = []
    out_list = []
    y_gt_list = []

    # process data
    g = get_graph_from_grahml(graph_file, "state")

    # good or useless
    n_nodes = g.number_of_nodes()

    # Maybe we can use this directly
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
    adj = nx.to_scipy_sparse_matrix(g).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    jnt_coordinates = np.array([], dtype=np.float64)

    xyz = None
    for node in g.nodes:
        coordinates = g.nodes[node]['state']
        xyz = np.array(coordinates, dtype=np.float64)
        jnt_coordinates = np.append(jnt_coordinates, xyz)

    jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])

    data = Data(x=torch.from_numpy(jnt_coordinates).double(), edge_index=edge_index)

    with torch.no_grad():
        cuda_data = data.to(device)
        cuda_data.x = cuda_data.x/torch.pi
        out = model(cuda_data)
        if visual_type == "result":
            out = torch.round(out)

    out = out.cpu().detach().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 14))
    from torch_geometric.utils.convert import to_networkx
    len_data = data.x.shape[0]
    g = to_networkx(data, to_undirected=True)
    data_x = data.x.cpu().detach().numpy()*torch.pi

    show_edges = True
    if show_edges:
        edge_colors = ["black"] * g.number_of_edges()
        nx.draw_networkx_edges(
            g, data_x, alpha=0.4, ax=axs, edge_color=edge_colors)

    node_color = ["blue"] * len_data
    nc = nx.draw_networkx_nodes(
        g,
        data_x,
        nodelist=g.nodes,
        node_size=60,
        cmap=plt.cm.bwr_r,
        node_color=out.tolist(),
        vmin=0,
        vmax=1.0,
        ax=axs
    )
    axs.set_aspect('equal', adjustable='box')
    axs.set_xlim(-3.5, 3.5)
    axs.set_ylim(-3.5, 3.5)  # slightly greater than pi
    fig.colorbar(nc)
    plt.show()
    return out


def visualize_graph_classification(model_file, graph_file, dataset_type):
    # Load data
    cwd = os.getcwd()
    if cwd not in graph_file:
        graph_file = os.path.join(cwd, graph_file)
    if dataset_type == "local":
        name = "removal_partial_labelled_local_test"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalPartialLabelledDatasetLocal(root=data_root, hop=2, n_files=50, dataset_name=name, training=False)
        model = GCNwLabels()
    else:
        name = "removal_global_test"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalDatasetGlobal(root=data_root, dataset_name=name, training=False)
        model = GCN()
    data_loader = DataLoader(dataset, batch_size=1)

    # Define the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(cwd, model_file)
    model.load_state_dict(torch.load(model_path))

    model.double()
    model = model.to(device)

    # process data
    g = get_graph_from_grahml(graph_file, "state")

    n_nodes = g.number_of_nodes()
    out_list = []
    y_gt_list = []

    with torch.no_grad():
        model.eval()
        for node in data_loader:
            node = node.to(device)
            out = model(node)
            if dataset_type == "local":
                out_list.append(out[node.test_mask].cpu().detach().numpy())
                y_gt_list.append(node.y[node.test_mask].cpu().detach().numpy())
            else:
                out_list.append(out.cpu().detach().numpy())
                y_gt_list.append(node.y.cpu().detach().numpy())

    if dataset_type == "local":
        out_list = [np.exp(a.squeeze()) for a in out_list]
        # Careful! Label 0 is important, Label 1 is useless! Revert the label
        pred_list = [np.argmax(a) for a in out_list]

        # Now Label 1 is important, Label 0 is useless
        pred_list = [1 if a==0 else 0 for a in pred_list]
        y_gt_list = [1 if a.squeeze().tolist() == [1, 0] else 0 for a in y_gt_list]
        out_list = [a[0] for a in out_list]
    else:
        out_list = out_list[0].squeeze().tolist()
        pred_list = [round(a) for a in out_list]
        y_gt_list = y_gt_list[0].squeeze().tolist()

    # Prepare plot
    results_list = [pred_list, y_gt_list, out_list]
    plot_titles = ["Prediction", "Ground truth", "Output"]
    right_pred = [1 if pred_list[i] == y_gt_list[i] else 0 for i in range(len(y_gt_list))]
    right_posivtive_pred = [1 if pred_list[i]==1 and y_gt_list[i]==1 else 0 for i in range(len(pred_list))]

    acc = sum(right_pred)/len(right_pred)
    print("ACC", acc)
    recall = sum(right_posivtive_pred)/sum(y_gt_list)
    print("Recall", recall)

    # nx.spring_layout(G)
    node_states = [g.nodes[i]["state"] for i in range(n_nodes)]
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(50, 10))

    for i in range(3):
        print(results_list[i])
        show_edges = True
        if show_edges:
            edge_colors = ["black"] * g.number_of_edges()
            nx.draw_networkx_edges(
                g, node_states, alpha=0.4, ax=axs[i], edge_color=edge_colors)
        #
        nc = nx.draw_networkx_nodes(
            g,
            node_states,
            nodelist=g.nodes,
            node_size=60,
            cmap=plt.cm.bwr_r,
            node_color=results_list[i],
            ax=axs[i],
            vmin=0,
            vmax=1
        )
        # https: // matplotlib.org / stable / tutorials / colors / colormaps.html
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].set_xlim(-3.5, 3.5)
        axs[i].set_ylim(-3.5, 3.5)  # slightly greater than pi
        axs[i].set_title(plot_titles[i])

    color_list = ["blue" if a==1 else "white" for a in y_gt_list]
    nc = nx.draw_networkx_nodes(
        g,
        node_states,
        nodelist=g.nodes,
        node_size=60,
        #cmap=plt.cm.bwr_r,
        node_color=color_list,
        ax=axs[3],
        vmin=0,
        vmax=1
    )
    # https: // matplotlib.org / stable / tutorials / colors / colormaps.html
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].set_xlim(-3.5, 3.5)
    axs[3].set_ylim(-3.5, 3.5)  # slightly greater than pi
    axs[3].set_title("Important only")

    #plt.colorbar(nc)
    plt.suptitle(dataset_type + model_file)
    plt.show()
    return out_list


def graph_classification_main(model_file, local=True, visual_type="result"):
    model_file = "../data/models/gnn_node_classifier_1669213081038970648.pt"
    if local:
        out_list = visualize_graph_classification(model_file, visual_type="p")
    else:
        out_list = visualize_global_graph_classification(model_file, visual_type="result")
    cwd = os.getcwd()
    graph_file = cwd + "/../data/graphs/graph_250_nodes_99.graphml"

    # process data
    n_node = 250
    easy_pl_req_path = cwd + "/../data/pl_req/easy_pl_req_{}_nodes.json".format(n_node)
    file_name = cwd + "/../data/compare/graph_250_nodes_99_20-80.json"
    g = get_graph_from_grahml(graph_file, "state")

    # sorted_out_list = sorted(out_list)
    # print(sorted_out_list)
    # threshold = round(len(sorted_out_list)*0.1)
    # threshold = sorted_out_list[threshold]
    # out_list = [True if out > threshold else False for out in out_list]

    threshold = 0.2
    out_list = [True if out > threshold else False for out in out_list]

    planning_req_list = load_planning_requests(easy_pl_req_path)
    compare_planning_performance_with_classification(g, out_list, planning_req_list, file_name)


def compare_planning_performance_with_classification(g, labels, planning_reqs, file_path,
                                                     results_before_removing=None):
    """
    Remove the nodes that classified as useless,
    save the planning performance in terms of success rate, checking counts and path length
    """
    dof = 2
    links = [0.5, 0.5]
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    pl_env.G = g
    from joblib import Parallel, delayed
    n_planning_req = len(planning_reqs)
    n_jobs = 12
    if results_before_removing is None:
        print("Planning before removing....")
        results_before_removing = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
            planning_reqs[i]) for i in range(n_planning_req))

    for i, label in enumerate(labels):
        if label < 0.5:
            pl_env.G.remove_node(i)

    print("Planning after removing....")
    results_after_removing = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        planning_reqs[i]) for i in range(n_planning_req))

    # Process the data
    before = process_results_list(results_before_removing)
    after = process_results_list(results_after_removing)

    before_after = {"before": before, "after": after}
    write_to_json(before_after, file_path)


def benchmarking_two_roadmaps(baseline_roadmap, improved_roadmap):
    # Load file
    g_baseline = baseline_roadmap  # get_graph_from_grahml(baseline_roadmap, "state")
    g_improved = improved_roadmap  # get_graph_from_grahml(improved_roadmap, "state")
    planning_request_file = ["../data/pl_req/easy_pl_req_250_nodes.json",
                             "../data/pl_req/hard_pl_req_250_nodes.json"]

    # Define planning environment or planning agnent
    dof = 2
    links = [0.5, 0.5]
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)

    # Baseline
    print("Computing baseline...")
    pl_env.G = deepcopy(g_baseline)
    r_baseline, cr_baseline = planning_through_request_dataset(
        planning_request_file, planning_agent=pl_env)

    # Imporved
    print("Computing improved graph...")
    pl_env.G = deepcopy(g_improved)
    r_improved, cr_improved = planning_through_request_dataset(
        planning_request_file, planning_agent=pl_env)

    # Output data
    # Success rate, path length, checking counts, list of improvement, list of worst results
    n_data = len(cr_baseline["success"])
    success_rate_baseline = sum(cr_baseline["success"]) / n_data
    success_rate_improved = sum(cr_improved["success"]) / n_data

    checked_cnt_baseline = sum(cr_baseline["checked_counts"])
    checked_cnt_improved = sum(cr_improved["checked_counts"])

    adjusted_success = [cr_improved["success"][i] and cr_baseline["success"][i] for i in range(n_data)]
    path_length_baseline = sum([cr_baseline["path_length"][i]*adjusted_success[i] for i in range(n_data)])
    path_length_improved = sum([cr_improved["path_length"][i]*adjusted_success[i] for i in range(n_data)])

    # Success Improved list
    success_better = []
    success_worse = []
    for i in range(n_data):
        if cr_improved["success"][i] and not cr_baseline["success"][i]:
            success_better.append(i)
        elif not cr_improved["success"][i] and cr_baseline["success"][i]:
            success_worse.append(i)

    print(f"Success rate   - before (baseline) {success_rate_baseline} - after {success_rate_improved}")
    print(f"Checked counts - before (baseline) {checked_cnt_baseline} - after {checked_cnt_improved}")
    print(f"Path length    - before (baseline) {path_length_baseline} - after {path_length_improved}")

    # Write data


def planning_through_request_dataset(list_request_file, planning_agent):
    # Load planning request
    requests = []
    requests_mark = [0]
    for request_file in list_request_file:
        requests = requests + load_planning_requests(request_file)
        requests_mark.append(len(requests))

    requests = requests[-1000:]
    # Plan
    n_jobs = 14
    pr_list = Parallel(n_jobs=n_jobs)(delayed(planning_agent.search)(
        req) for req in requests)

    # Process the planning results
    success = [pr.has_solution for pr in pr_list]
    checked_counts = [pr.checked_counts for pr in pr_list]
    solution_path = [pr.solution_path for pr in pr_list]

    path_length = []
    for s_path in solution_path:
        if s_path:
            diffs = [s_path[i + 1] - s_path[i] for i in range(len(s_path) - 1)]
            path_len = sum([np.linalg.norm(diff) for diff in diffs])
        else:
            path_len = 0
        path_length.append(path_len)

    # Return result in dictionary
    # TODO: not sure if we should save it in this format
    results = {request_file: {"success": success[requests_mark[i]:requests_mark[i+1]],
                              "checked_counts": checked_counts[requests_mark[i]:requests_mark[i+1]],
                              "path_length": path_length[requests_mark[i]:requests_mark[i+1]]}
               for i, request_file in enumerate(list_request_file)}

    results_combined_view = {"success": success,
                             "checked_counts": checked_counts,
                             "path_length": path_length}

    return results, results_combined_view


if __name__ == "__main__":
    # cwd = os.getcwd()
    # file_name_basis = cwd + "/../data/compare/graph_250_nodes_{}_after_selected_removal.json"
    # # results_before_removing = {"success_sum": 4754,
    # #                        "checked_counts": 4637502,
    # #                        "path_length": 27180.793973936416}
    # usage_file = cwd + "/../data/nodes/graph_250_nodes_usage_{}.json"
    # visualize_graph_ground_truth(file_name_basis.format(1))

    local_model_path = "/home/huang/ml_workspace/mp2d/data/models/local_gnn_node_classifier_40_acc_65_67.pt"
    global_model_path = "../data/models/global_gnn_node_classifier_500_acc_70_68.pt"
    graph_file = "../data/graphs/graph_250_nodes_51.graphml"
    dataset_type = "local"
    # if dataset_type == "global":
    #     model_path = global_model_path
    # else:
    #     model_path = local_model_path
    # visualize_graph_classification(model_path, graph_file, dataset_type)
    data_file = "../data/compare/structured_graph_after_selected_removal.json"
    original_path = "../data/compare/structured_graph_paths.json"
    visualize_cost_data_file(data_file, original_path)

