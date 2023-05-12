import numpy as np

from planning import Planning, get_2dof_planning_environment
from manipulator import *
import os
from copy import deepcopy
from utilities import *
from joblib import Parallel, delayed
import time


def record_node_usage(config):
    """
    Iterate the planning requests and find a solution in the given graph.
    Record the nodes in the final solution.
    Write the usages of the nodes in the given json file.
    Support multiprocessing
    :param config - A configuration of all data paths needed including graph, planning request
    :return: None
    """
    pl_env = get_2dof_planning_environment()

    # Get planning request dataset and graph dataset
    planning_req_dataset_file = config["requests_file"]
    planning_req_list = load_planning_requests(planning_req_dataset_file)
    #planning_req_list = planning_req_list[:13]

    gf = config["graph_file"]
    print("Processing graph {}...".format(gf))
    g = get_graph_from_grahml(gf, "state")
    pl_env.G = deepcopy(g)
    pl_env.rewire_graph()
    n_nodes = pl_env.G.number_of_nodes()
    node_usage_cnts = {i: 0 for i in range(n_nodes)}
    node_usage_pl_req = {i: [] for i in range(n_nodes)}
    failure = 0
    success = 0

    for i, pl_req in enumerate(planning_req_list):
        pr = pl_env.search(pl_req)
        print(f"{i}/{len(planning_req_list)}")
        if pr.has_solution:
            success += 1
            idx_path = pl_env.idx_path
            for solution_idx in idx_path:
                if solution_idx < n_nodes:
                    # Record usage
                    node_usage_cnts[solution_idx] = node_usage_cnts[solution_idx] + 1
                    # Record planning request
                    node_usage_pl_req[solution_idx].append({planning_req_dataset_file: i})
        else:
            failure += 1
        # Write to file....

    json_dict = {
        "graph_file": gf,
        "success": success,
        "failure": failure,
        "node_usage_counts": node_usage_cnts,
        "node_used_for_req": node_usage_pl_req}
    node_usage_cnts_file_name = config["node_usage_file"]
    write_to_json(json_dict, node_usage_cnts_file_name)


def filter_out_important_samples(pl_env: Planning, planning_request: PlanningRequest, solution: list):
    """
    Find out the nodes that can make things different,
    meaning that one can not find a solution in the graph without this node
    solutions: list of numpy array
    """
    added_nodes = []
    important_nodes = []
    print("Filtering out important samples")
    n_nodes = pl_env.G.number_of_nodes()
    # Adding notes to the graph until there is a solution
    print(solution)
    for i, node in enumerate(solution):
        if isinstance(node, list):
            node = np.array(node)
        pl_env.add_node(node, max_nn_number=30)
        pr = pl_env.search(planning_request)
        added_nodes.append(n_nodes+i)
        if pr.has_solution:
            # get index path
            important_nodes_idx = list(set(pl_env.idx_path).intersection(added_nodes))
            important_nodes = [pl_env.G.nodes[idx]["state"] for idx in important_nodes_idx]
            print("Has solution")
            break

    for node_idx in added_nodes:
        pl_env.G.remove_node(node_idx)

    node_dict = {}
    # Remove the start and goal positions
    for node in important_nodes:
        dist_to_start = np.linalg.norm(node - planning_request.start)
        dist_to_goal = np.linalg.norm(node - planning_request.goal)

        if dist_to_start > 1e-6 and dist_to_goal > 1e-6:
            idx = pl_env.G.number_of_nodes() + 1000
            pl_env.add_node(node, idx=idx)
            node_dict.update({idx: node})

    # Removing nodes from the graph to see if we really need these nodes for a solution
    for node_idx, node in node_dict.items():
        pl_env.G.remove_node(node_idx)
        pr = pl_env.search(planning_request)
        pl_env.add_node(node, node_idx)
        if pr.has_solution:
            node_dict[node_idx] = False

    important_nodes = [x for x in node_dict.values() if x is not False]
    return important_nodes


# TODO: remove this function and use the one from multi_dof....
def generate_graphs(n_graphs: int, n_nodes: int, basic_file_name: str):
    """
    Generate a set af graphs with given numbers of nodes.
    Write them separately in .graphml format
    :param n_graphs:
    :param n_nodes:
    :param basic_file_name:
    :return:
    """
    pl_env = get_2dof_planning_environment()
    graphs_file_names = {}

    for i in range(n_graphs):
        pl_env.generate_graph_halton(n_nodes)
        pl_env.rewire_graph()
        file_name = basic_file_name + "_{}".format(i) + ".graphml"
        pl_env.save_graph_to_file(file_name)
        graphs_file_names.update({i: file_name})

    graph_file_collection_name = basic_file_name + "file_names.json"
    write_to_json(graphs_file_names, graph_file_collection_name)


# TODO: remove this function and use the one from multi_dof....
def generate_graphs_with_obstacles(obstacles: list, n_nodes: int, file_names: list):
    pl_env = get_2dof_planning_environment()
    graphs_file_names = {}

    n_graphs = len(obstacles)
    assert n_graphs == len(file_names)

    for i in range(n_graphs):
        pl_env.generate_graph_halton(n_nodes, obstacles=obstacles[i])
        pl_env.save_graph_to_file(file_names[i])
        graphs_file_names.update({i: file_names[i]})

    graph_file_collection_name = "../data/graphs/graph_with_obstacles_file_names.json"
    write_to_json(graphs_file_names, graph_file_collection_name)


def generate_planning_requests(n_data, n_nodes, hard_dataset_file_path, easy_dataset_file_path):
    """
    Generate a set of random planning requests and write them into two json files
    :param n_data:
    :param n_nodes:
    :param hard_dataset_file_path:
    :param easy_dataset_file_path:
    :return:
    """
    from rrt_connect import RRTConnect
    hard_planning_requests = []
    easy_planning_requests = []

    pl_env = get_2dof_planning_environment()
    pl_env.generate_graph_halton(n_nodes)

    for i in range(n_data):
        solved = False
        req = None

        print("Generating planning {}-th requests...".format(i))
        checked_list = []
        while (not solved) or len(checked_list) < 3:
            obstacles = generate_random_obstacles(min_number=3)
            start = pl_env.generate_valid_sample(obstacles)
            goal = pl_env.generate_valid_sample(obstacles)

            req = PlanningRequest(start, goal)
            req.obstacles = obstacles

            pl_alg = RRTConnect(pl_env)
            solved = pl_alg.planning(req)
            pl_time = pl_alg.get_planning_time()

            if not solved:
                print("No solution found in RRT!")
                continue

            if pl_time < 0.1:
                continue

            print("Trying to find solution using the graph...")
            pr = pl_env.search(req, visualize=False)

        if pr.has_solution:
            easy_planning_requests.append(req)
            print("Solution found in the graph search")
        else:
            hard_planning_requests.append(req)
            print("No solution found in the graph search!")
    print("---")

    write_planning_request_to_json(easy_dataset_file_path, easy_planning_requests)
    write_planning_request_to_json(hard_dataset_file_path, hard_planning_requests)


def generate_hard_planning_requests(n_datas, n_nodes, hard_dataset_file_path):
    """
    Only generate the planning request that no solution can be found in a graph
    (Solution may be found in other graphs)
    :param n_datas:
    :param n_nodes:
    :param hard_dataset_file_path:
    :return:
    """
    from rrt_connect import RRTConnect
    hard_planning_requests = []
    dof = 2
    links = [0.5, 0.5]
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    pl_env.generate_graph_halton(n_nodes)
    cnt = 0

    while cnt < n_datas:
        solved = False
        req = None

        checked_list = []
        while (not solved) or len(checked_list) < 3:
            obstacles = generate_random_obstacles(min_number=3)
            start = pl_env.generate_valid_sample(obstacles)
            goal = pl_env.generate_valid_sample(obstacles)

            req = PlanningRequest(start, goal)
            req.obstacles = obstacles

            pl_alg = RRTConnect(pl_env)
            solved = pl_alg.planning(req)
            pl_time = pl_alg.get_planning_time()

            # path = pl_alg.get_solution()
            # is_path_valid = pl_env.check_path_validity(path, req.obstacles)
            # if is_path_valid:
            #     pl_env.visualize_request(req)
            #     pl_env.visualize_path(req, path)
            # else:
            #     print("Path is not valid!!!")

            if not solved:
                print("No solution found in RRT!")
                continue

            if pl_time < 0.1:
                continue

            print("Trying to find solution using the graph...")
            pr = pl_env.search(req)

        if pr.has_solution:
            print("Solution found in the graph search")
        else:
            hard_planning_requests.append(req)
            print("No solution found in the graph search!")
    print("---")
    write_planning_request_to_json(hard_dataset_file_path, hard_planning_requests)


def generate_dataset_using_removal(planning_dataset_file,
                                   graph_file,
                                   save_file_basis,
                                   results_before_removing=None):
    """
    See difference for all planning request if a node is removed from the graph
    """

    dof = 2
    links = [0.5, 0.5]
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])
    planning_requests = load_planning_requests(planning_dataset_file)
    g = get_graph_from_grahml(graph_file, "state")

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    pl_env.G = g
    n_planning_req = len(planning_requests)
    n_jobs = 12
    if results_before_removing is None:
        print("Planning before removing....")
        pr_list = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
            planning_requests[i]) for i in range(n_planning_req))
        results_before_removing = get_result_statistics(process_results_list(pr_list))
        file_name = save_file_basis.format("before")
        write_to_json(results_before_removing, file_name)

    n_nodes = g.number_of_nodes()
    nodes_recording = {}

    for node_idx in range(n_nodes):
        print("Removing node {}....".format(node_idx))
        pl_env.G = deepcopy(g)
        # Get the sample before removal
        samples = [pl_env.G.nodes[int(node_idx)]["state"].tolist()]
        pl_env.G.remove_node(int(node_idx))

        pr_list = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
            req, idx_start=1000, idx_goal=1001) for req in planning_requests)

        results_after_removing = get_result_statistics(process_results_list(pr_list))

        node_is_useless = True
        if results_after_removing["success_sum"] < results_before_removing["success_sum"]:
            node_is_useless = False
        elif results_after_removing["checked_counts"] > results_before_removing["checked_counts"]*1.1:
            node_is_useless = False

        data_dict = {"nodes": samples, "graph": graph_file, "index": node_idx,
                     "character": "useless" if node_is_useless else "important",
                     "statistics": results_after_removing}
        print(data_dict)
        nodes_recording.update({node_idx: data_dict})

    file_name = save_file_basis.format("after")
    write_to_json(nodes_recording, file_name)


def generate_cost_dataset_using_removal_from_path(node_usage_file,
                                                  save_file_basis):
    """
    See extra cost for the planning requests if the nodes in the final paths for these requests
    are removed from the graph
    """
    print(node_usage_file)
    # Load data
    nodes_usage = read_from_json(node_usage_file)
    graph_file = fix_file_path(nodes_usage["graph_file"])
    g = get_graph_from_grahml(graph_file, "state")
    n_nodes = g.number_of_nodes()

    # Get the file of planning requests
    for i in range(n_nodes):
        reqs_list_for_first_node = list(nodes_usage["node_used_for_req"].values())[i]
        if reqs_list_for_first_node:
            break

    # Get planning requests
    planning_request_file = list(reqs_list_for_first_node[0].keys())[0]
    planning_request_file = fix_file_path(planning_request_file)
    planning_requests = load_planning_requests(planning_request_file)

    # Initialize the robot model
    dof = 2
    links = [0.5, 0.5]
    ma = manipulator(dof, links)
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    nodes_recording = {}

    n_jobs_max = 14
    # Update the nodes_recoding
    starting_time = time.time_ns()
    for node_idx in range(n_nodes):
        # Print the time used
        diff = time.time_ns() - starting_time
        print("Time difference", diff/1e9)
        starting_time = time.time_ns()

        # Skip the ones without being on a single planning results
        # The node with no usages are directly useless
        samples = g.nodes[node_idx]["state"]
        if nodes_usage["node_usage_counts"][str(node_idx)] < 1:
            data_dict = {"nodes": samples.tolist(),
                         "graph": graph_file,
                         "index": node_idx,
                         "character": "useless"}
            nodes_recording.update({node_idx: data_dict})
            continue
        pl_env.G = deepcopy(g)

        # Get planning requests that has this node_idx in the final solution
        reqs_list_for_this_node = nodes_usage["node_used_for_req"][str(node_idx)]
        req_idx_list = [list(d.values())[0] for d in reqs_list_for_this_node]

        # Find solutions in parallel
        n_jobs = min(n_jobs_max, len(req_idx_list))
        pr_before = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
            planning_requests[req_idx], idx_start=1000, idx_goal=1001) for req_idx in req_idx_list)

        # Remove the node in the graph and compare the planning results
        pl_env.G.remove_node(node_idx)
        pr_after = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
                planning_requests[req_idx], idx_start=1000, idx_goal=1001)
                                                   for req_idx in req_idx_list)

        pr_before_stats = get_result_statistics(process_results_list(pr_before))
        pr_after_stats = get_result_statistics(process_results_list(pr_after))
        checked_counts_diff = pr_after_stats["checked_counts"] - pr_before_stats["checked_counts"]
        success_sum_diff = pr_after_stats["success_sum"] - pr_before_stats["success_sum"]
        path_length_sum_diff = pr_after_stats["path_length_sum"] - pr_before_stats["path_length_sum"]

        # A list of nodes that have been visited
        visited_nodes_before = pr_before_stats["visited_nodes"]
        visited_nodes_after = pr_after_stats["visited_nodes"]

        # A dictionary of nodes and the frequency of blocked
        blocked_nodes_before = pr_before_stats["blocked_nodes"]
        blocked_nodes_after = pr_after_stats["blocked_nodes"]

        # A dictionary of planning request and solution paths
        path_list_before = dict(zip(req_idx_list, pr_before_stats["path_list"]))
        path_list_after = dict(zip(req_idx_list, pr_after_stats["path_list"]))

        index_path_list_before = dict(zip(req_idx_list, pr_before_stats["index_path_list"]))
        index_path_list_after = dict(zip(req_idx_list, pr_after_stats["index_path_list"]))

        character = "not-clear"
        data_dict = {"nodes": samples.tolist(),
                     "graph": graph_file,
                     "requests": planning_request_file,
                     "index": node_idx,
                     "character": character,
                     "checked_counts_diff": checked_counts_diff,
                     "success_sum_diff": success_sum_diff,
                     "path_length_sum_diff": path_length_sum_diff,
                     "checked_counts_before": pr_before_stats["checked_counts"],
                     "checked_counts_after": pr_after_stats["checked_counts"],
                     "success_sum_before": pr_before_stats["success_sum"],
                     "success_sum_after": pr_after_stats["success_sum"],
                     "path_length_sum_before": pr_before_stats["path_length_sum"],
                     "path_length_sum_after": pr_after_stats["path_length_sum"],
                     "path_list_before": path_list_before,
                     "path_list_after": path_list_after,
                     "index_path_list_before": index_path_list_before,
                     "index_path_list_after": index_path_list_after,
                     "visited_nodes_before": visited_nodes_before,
                     "visited_nodes_after": visited_nodes_after,
                     "blocked_nodes_before": blocked_nodes_before,
                     "blocked_nodes_after": blocked_nodes_after,
                     }

        nodes_recording.update({node_idx: data_dict})
        print(data_dict)

    file_name = save_file_basis.format("after_selected")
    write_to_json(nodes_recording, file_name)


def record_original_path(config):
    """
    Record path for each planning request with and without obstacles
    Additional information:
        - node usages without obstacles
        - node usages with obstacles
        - path information for RRTConnect
    """
    # Load data
    gf = fix_file_path(config["graph_file"])
    g = get_graph_from_grahml(gf, "state")
    n_nodes = g.number_of_nodes()

    # Get planning requests
    planning_request_file = config["requests_file"]
    planning_request_file = fix_file_path(planning_request_file)
    planning_requests = load_planning_requests(planning_request_file)
    # planning_requests = planning_requests[0:13]
    planning_requests_wo_obstacles = deepcopy(planning_requests)
    for req in planning_requests_wo_obstacles:
        req.obstacles = []

    # Initialize the robot model
    pl_env = get_2dof_planning_environment()
    nodes_recording = {}

    pl_env.G = deepcopy(g)

    # Find solutions in parallel using Graph
    n_jobs = 14
    pr_with_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=1000, idx_goal=1001) for req in planning_requests)

    pr_without_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=1000, idx_goal=1001) for req in planning_requests_wo_obstacles)

    # Find solutions in parallel using RRT
    def plan_with_rrt_connect(pl_req):
        pl_alg = RRTConnect(pl_env)
        solved = pl_alg.planning(pl_req)
        pr_rrt_connect = PlanningResult()
        if solved is False:
            pr_rrt_connect.has_solution = False
            pr_rrt_connect.solution_path = []
            return pr_rrt_connect

        solution_path = pl_alg.get_solution()
        is_path_valid = pl_env.check_path_validity(solution_path, pl_req.obstacles)
        if not is_path_valid:
            pr_rrt_connect.has_solution = False
            pr_rrt_connect.solution_path = []
            return pr_rrt_connect

        pr_rrt_connect.has_solution = True
        pr_rrt_connect.solution_path = solution_path
        return pr_rrt_connect

    pr_rrt_connect_with_obs = Parallel(n_jobs=n_jobs)(delayed(plan_with_rrt_connect)(
        req) for req in planning_requests)

    pr_rrt_connect_without_obs = Parallel(n_jobs=n_jobs)(delayed(plan_with_rrt_connect)(
        req) for req in planning_requests_wo_obstacles)

    # Process the planning results
    def process_results(pr_list):
        nodes_idx = list(range(n_nodes))
        zero_usages = [0]*n_nodes
        node_usages = dict(zip(nodes_idx, zero_usages))
        index_solution_path_dict = {}
        for pr_id, pr in enumerate(pr_list):
            pr.index_path = pr.index_path[1:-1]
            index_solution_path_dict[pr_id] = pr.index_path
            for idx in pr.index_path:
                node_usages[int(idx)] = node_usages[int(idx)] + 1
        return index_solution_path_dict, node_usages

    index_solution_path_with_obs, node_usages_with_obs = process_results(pr_with_obs)
    index_solution_path_without_obs, node_usages_without_obs = process_results(pr_without_obs)

    pl_req_ids = index_solution_path_with_obs.keys()
    index_solution_paths = {}
    for req_id in pl_req_ids:
        rrt_sp_with_obs = [node.tolist() for node in pr_rrt_connect_with_obs[req_id].solution_path[1:-1]]
        rrt_sp_without_obs = [node.tolist() for node in pr_rrt_connect_without_obs[req_id].solution_path[1:-1]]

        data_dict = {"start": planning_requests[req_id].start.tolist(),
                     "goal": planning_requests[req_id].goal.tolist(),
                     "with_obstacles": index_solution_path_with_obs[req_id],
                     "without_obstacles": index_solution_path_without_obs[req_id],
                     "rrt_solution_with_obstacles": rrt_sp_with_obs,
                     "rrt_solution_without_obstacles": rrt_sp_without_obs,
                     "with_obstacles_checked_nodes": pr_with_obs[req_id].checked_dict}

        index_solution_paths.update({req_id: data_dict})

    data_to_write = {"graph_file": gf,
                     "requests_file": planning_request_file,
                     "index_solution_paths": index_solution_paths,
                     "node_usages_with_obstacles": node_usages_with_obs,
                     "node_usages_without_obstacles": node_usages_without_obs}

    # TODO: ADD request & graph file
    save_file = config["original_path_file"]
    write_to_json(data_to_write, save_file)


def process_original_path_data_file(data_file, proccessed_data_file):
    """
    If a node or its edge on the solution is blocked by dynamic obstacles,
    find out its plan B nodes.

    Conditions:
    1. The node in the original path (without dynamic obstacles) is checked by algorithm
    (The algorithm might not even reach some nodes in the original path
    because the previous one already blocked )
    2. The node OR the edge is blocked

    Plan B:
    - Difference between the new path and the original path.

    TODOs:
    - We may need more exact filtering about the plan B nodes.

    """
    # Load data
    data = read_from_json(data_file)
    gf = data["graph_file"]
    g = get_graph_from_grahml(gf, "state")

    # Get mapping from node index in the graph to planning requests - withOUT obstacles
    n_nodes = g.number_of_nodes()
    node_request_map = dict(zip(list(range(n_nodes)), [[]]*n_nodes))
    node_planB_map = dict(zip(list(range(n_nodes)), [[]]*n_nodes))

    index_solution_paths = data["index_solution_paths"]
    for req_id, data_pt in index_solution_paths.items():
        idx_path_without_obstacles = data_pt["without_obstacles"]
        for node_id in idx_path_without_obstacles:
            # Update the request to the mapping
            node_request_map[node_id].append(req_id)

            # Mapping from blocked nodes to Plan-B nodes
            checked_nodes = data_pt["with_obstacles_checked_nodes"]  # Get which nodes are checked
            if str(node_id) in checked_nodes:
                if checked_nodes[str(node_id)] is False:
                    # False meaning the node is blocked OR edge is block..
                    # To find out the exact blocked position - Use this
                    # - record_blocked_position_and_plan_bs
                    # The plan-B nodes should be the difference between two paths TODO: Is it correct?
                    plan_b = list(set(data_pt["with_obstacles"]) - set(idx_path_without_obstacles))
                    # TODO filter out the Plan-Bs
                    node_planB_map[node_id] = node_planB_map[node_id] + plan_b

    processed_data = {}
    for node_id in range(n_nodes):
        planB_states = [g.nodes[i]["state"].tolist() for i in node_planB_map[node_id]]
        processed_data[node_id] = {"request": node_request_map[node_id],
                                   "planB_nodes": node_planB_map[node_id],
                                   "planB_node_states": planB_states}

    write_to_json(processed_data, proccessed_data_file)


def record_blocked_position_and_plan_bs(config):
    """
    Record where in the original path is blocked.
    """

    # Get config
    save_file = config["processed_plan_b_file"]
    gf = config["graph_file"]
    planning_request_file = config["requests_file"]
    # TODO extend this part to save computation
    original_path_file = config["original_path_file"]

    # Try to load the data from original path
    has_original_path_data = os.path.isfile(original_path_file)
    if has_original_path_data:
        original_path_data = read_from_json(original_path_file)

    # Load data
    gf = fix_file_path(gf)
    g = get_graph_from_grahml(gf, "state")

    # Get planning requests
    planning_request_file = fix_file_path(planning_request_file)
    planning_requests = load_planning_requests(planning_request_file)
    # planning_requests = planning_requests[0:13]
    planning_requests_wo_obstacles = deepcopy(planning_requests)
    for req in planning_requests_wo_obstacles:
        req.obstacles = []

    # Initialize the robot model
    pl_env = get_2dof_planning_environment()
    pl_env.G = deepcopy(g)

    # Find solutions in parallel using Graph
    if not has_original_path_data:
        n_jobs = 14
        pr_without_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=1000, idx_goal=1001) for req in planning_requests_wo_obstacles)
        pr_with_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=1000, idx_goal=1001) for req in planning_requests)
    else:
        pr_without_obs = []
        pr_with_obs = []
        index_solution_paths = original_path_data["index_solution_paths"]
        for _, index_solution in index_solution_paths.items():
            pr_w = PlanningResult()
            pr_wo = PlanningResult()
            pr_w.index_path = index_solution["with_obstacles"]
            pr_wo.index_path = index_solution["without_obstacles"]

            pr_w.has_solution = True if pr_w.index_path else False
            pr_wo.has_solution = True if pr_wo.index_path else False

            pr_w.solution_path = [g.nodes[idx]["state"] for idx in pr_w.index_path]
            pr_wo.solution_path = [g.nodes[idx]["state"] for idx in pr_wo.index_path]

            pr_without_obs.append(pr_wo)
            pr_with_obs.append(pr_w)

    # Process the planning results - We want to compute the real blocked positions
    # Finding out the pairs of blocked and plan-b nodes
    req_blocked_nodes_dict = {}
    req_path_diff_dict = {}
    for pr_id, pr in enumerate(pr_without_obs):
        # Skip to next planning result if no solution is found
        if not pr_with_obs[pr_id].has_solution:
            continue

        solution_without_obs = [planning_requests[pr_id].start] + \
                               pr.solution_path[1:-1] + \
                               [planning_requests[pr_id].goal]

        blocked_node_list = []
        for id_n in range(len(solution_without_obs) - 1):
            node = solution_without_obs[id_n]
            next_node = solution_without_obs[id_n + 1]
            is_valid, n_checking = pl_env.check_validity(
                node, next_node, planning_requests[pr_id].obstacles)
            if not is_valid:
                if n_checking == 1:
                    # In this case, next node is in collision - just skip
                    # next iteration will add this node into the blocked list
                    continue
                else:
                    diff = np.linalg.norm(next_node - node)
                    normalized_vector = (next_node - node) / diff
                    blocked_node = (n_checking - 2) * pl_env.resolution * normalized_vector + node
                    blocked_node_list.append(blocked_node)
        req_blocked_nodes_dict[pr_id] = blocked_node_list
        path_diff = list(set(pr_with_obs[pr_id].index_path) - set(pr.index_path))
        req_path_diff_dict[pr_id] = path_diff

    blocked_node_plan_b_list = []
    plan_b_blocked_node_dict = {}

    for req_id, blocked_list in req_blocked_nodes_dict.items():
        print(blocked_list)
        for blocked_node in blocked_list:
            path_diff_state = [g.nodes[i_n]["state"].tolist() for i_n in req_path_diff_dict[req_id]]
            path_diff_index = req_path_diff_dict[req_id]
            data_dict = {"blocked_node": blocked_node.tolist(),
                         "plan-B": path_diff_state}
            blocked_node_plan_b_list.append(data_dict)
            for idx in path_diff_index:
                plan_b_blocked_node_dict[idx] = plan_b_blocked_node_dict.get(idx, []) + [blocked_node.tolist()]

    data_to_write = {"graph_file": gf,
                     "requests_file": planning_request_file,
                     "plan_b_blocked_node_dict": plan_b_blocked_node_dict,
                     "blocked_node_plan_b_list": blocked_node_plan_b_list}
    write_to_json(data_to_write, save_file)