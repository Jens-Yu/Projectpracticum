from copy import deepcopy
from joblib import Parallel, delayed
from planning import *


# Generate robot models
def get_3dof_planning_environment():
    dof = 3
    links = [0.3]*dof
    planning_range_max = np.array([np.pi]*dof)
    planning_range_min = np.array([np.pi]*dof)

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    return pl_env


def get_7dof_planning_environment():
    dof = 7
    links = [0.15]*dof
    planning_range_max = np.array([np.pi]*dof)
    planning_range_min = np.array([-np.pi]*dof)

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    return pl_env


# Generate graphs with or without obstacles
def generate_graph(pl_env: Planning, n_nodes: int, graph_file: str, obstacles=None,
                   rewire_after_generation=False):
    """
    Generate a set af graphs with given numbers of nodes.
    Write them separately in .graphml format
    :param pl_env: environment that defines robot and static obstacles
    :param n_nodes:
    :param graph_file: The path to save the graph
    :param obstacles:
    :param rewire_after_generation: option to rewire graph to correct edge connections
    :return:
    """
    pl_env.generate_graph_halton(n_nodes, obstacles=obstacles)
    if rewire_after_generation:
        pl_env.rewire_graph()
    pl_env.save_graph_to_file(graph_file)


# Generate planning request
def generate_planning_requests(config, pl_env: Planning):
    """
    Generate a set of random planning requests and write them into two json files

    :param n_data: Number of planning requests that we want to generate
    :param pl_env: The planning environment where the robot, graph and static environment are defined
    :param hard_dataset_file: Path the save the hard planning request - No path in the given graph
    :param easy_dataset_file: Path the save the easy planning request
    :return:
    """
    from rrt_connect import RRTConnect
    # Load config
    n_data = config["n_data"]
    hard_requests_file_file = config["hard_requests_file"]
    easy_requests_file = config["easy_requests_file"]
    hard_rrt_solution_file = config["hard_rrt_solution_file"]
    easy_rrt_solution_file = config["easy_rrt_solution_file"]

    hard_planning_requests = []
    easy_planning_requests = []

    hard_planning_requests_rrt_solution = {}
    easy_planning_requests_rrt_solution = {}

    n_hard_requests = 0
    n_easy_requests = 0

    for i in range(n_data):
        solved = False
        req = None

        print(f"Generating planning for {pl_env.dof} problem {i}-th requests...")
        n_check_nodes = 0
        rrt_solution = None
        while (not solved) or n_check_nodes <= 3:
            obstacles = generate_random_obstacles(min_number=3)
            start = pl_env.generate_valid_sample(obstacles)
            goal = pl_env.generate_valid_sample(obstacles)

            req = PlanningRequest(start, goal)
            req.obstacles = obstacles

            pl_alg = RRTConnect(pl_env)
            solved = pl_alg.planning(req)  # Default solving time is 5 seconds
            pl_time = pl_alg.get_planning_time()

            if not solved:
                print("No solution found in RRT!")
                continue

            if pl_time < 0.1:
                continue

            print("Trying to find solution using the graph...")
            rrt_solution = pl_alg.get_solution()
            rrt_solution_list = [node.tolist() for node in rrt_solution]
            pr = pl_env.search(req, visualize=False)
            n_check_nodes = len(pr.checked_dict)

        if pr.has_solution:
            easy_planning_requests.append(req)
            easy_planning_requests_rrt_solution[n_easy_requests] = rrt_solution_list
            n_easy_requests += 1
            print("Solution found in the graph search")
        else:
            hard_planning_requests.append(req)
            hard_planning_requests_rrt_solution[n_hard_requests] = rrt_solution_list
            n_hard_requests += 1
            print("No solution found in the graph search!")
    print("---")

    write_planning_request_to_json(easy_requests_file, easy_planning_requests)
    write_planning_request_to_json(hard_requests_file, hard_planning_requests)

    # Write RRT solution to files
    write_to_json(hard_planning_requests_rrt_solution, hard_rrt_solution_file)
    write_to_json(easy_planning_requests_rrt_solution, easy_rrt_solution_file)


# Generate planning results & Block nodes
def record_blocked_position_plan_b_mapping(config, pl_env):
    """
    Record where in the original path is blocked.
    """
    # Get config
    save_file = config["processed_plan_b_file"]
    gf = config["graph_file"]
    planning_request_file = config["requests_file"]

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

    # Load graph into the planning environment
    assert pl_env.dof == np.shape(g.nodes[0]["state"])[0]
    pl_env.G = deepcopy(g)

    # Find solutions in parallel using Graph
    n_jobs = 14
    idx_start = 1000 + g.number_of_nodes()
    idx_goal = 1001 + g.number_of_nodes()
    pr_without_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=idx_start, idx_goal=idx_goal) for req in planning_requests_wo_obstacles)
    pr_with_obs = Parallel(n_jobs=n_jobs)(delayed(pl_env.search)(
        req, idx_start=idx_start, idx_goal=idx_goal) for req in planning_requests)

    index_solution_path_with_obs, node_usages_with_obs = process_results(pr_with_obs, n_nodes)
    index_solution_path_without_obs, node_usages_without_obs = process_results(pr_without_obs, n_nodes)

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

    plan_b_blocked_node_dict = {}
    req_to_mapping = {}

    for req_id, blocked_list in req_blocked_nodes_dict.items():
        data_dict_list = []
        for blocked_node in blocked_list:
            path_diff_state = [g.nodes[i_n]["state"].tolist() for i_n in req_path_diff_dict[req_id]]
            path_diff_index = req_path_diff_dict[req_id]
            data_dict = {"blocked_node": blocked_node.tolist(),
                         "plan-B": path_diff_state}
            # TODO - Should we add the request id here?
            data_dict_list.append(data_dict)
            for idx in path_diff_index:
                plan_b_blocked_node_dict[idx] = plan_b_blocked_node_dict.get(idx, []) + [blocked_node.tolist()]
        req_to_mapping[req_id] = data_dict_list

    list_of_attrs = ["request_to_mapping",
                     "plan_b_to_blocked_node",
                     "index_solution_path_with_obstacles",
                     "index_solution_path_without_obstacles",
                     "node_usages_explanation",
                     "node_usages_with_obstacles",
                     "node_usages_without_obstacles"]

    data_to_write = {"graph_file": gf,
                     "requests_file": planning_request_file,
                     "attributes": list_of_attrs,
                     "request_to_mapping_explanation": "Key: Request index, Value: blocked node and plan-B nodes",
                     "request_to_mapping": req_to_mapping,
                     "plan_b_blocked_node_explanation": "Key: Plan-B node index, Value: blocked node states",
                     "plan_b_to_blocked_node": plan_b_blocked_node_dict,
                     "index_solution_path_explanation": "Key: Request index, Value: Index solution path",
                     "index_solution_path_with_obstacles": index_solution_path_with_obs,
                     "index_solution_path_without_obstacles": index_solution_path_without_obs,
                     "node_usages_explanation": "Key: Node index, Value: Number of times the node is used in the plan",
                     "node_usages_with_obstacles": node_usages_with_obs,
                     "node_usages_without_obstacles": node_usages_without_obs
                     }

    write_to_json(data_to_write, save_file)


def record_frequency_with_obstacles(config, pl_env: Planning):
    # Get configurations
    planning_request_file = config["requests_file"]
    save_file = config["node_frequency_file"]
    number_of_requests = config["number_of_requests"]
    number_of_nodes = config["number_of_nodes"]
    number_of_static_obstacles = config["number_of_static_obstacles"]

    # Load data
    requests = load_planning_requests(planning_request_file)
    end_index_request = len(requests) - 1

    # Use the obstacles in one of the requests to generate the graph
    obstacles_list = [req.obstacles for req in requests[-number_of_static_obstacles:]]
    obstacles_list.reverse()

    # Limit the number of requests
    number_of_requests = min(number_of_requests, len(requests))
    requests = requests[0:number_of_requests]

    for i_obs, obstacles in enumerate(obstacles_list):
        print(f"Using the Obstacles from the end backwards {i_obs}")
        pl_env.generate_graph_halton(number_of_nodes, obstacles)
        index_start = 1000 + number_of_nodes
        index_goal = 1001 + number_of_nodes
        pr_list = Parallel(n_jobs=8)(delayed(pl_env.search)(
            req, idx_start=index_start, idx_goal=index_goal) for req in requests)

        # Get the frequency of each node
        index_solution_path_with_obs, node_usages_with_obs = process_results(
            pr_list, number_of_nodes)
        node_usages_with_obs = {k: {"counts": v, "state": pl_env.G.nodes[k]["state"].tolist()}
                                for k, v in node_usages_with_obs.items()}

        # Save the data
        write_to_json(node_usages_with_obs, save_file + f"_{end_index_request - i_obs}.json")


# Utils
def process_results(pr_list, n_nodes):
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


if __name__ == "__main__":
    # Config
    n_graphs = 5
    dof = 7

    n_nodes = 1000
    if dof == 7:
        n_nodes = 20000

    # Planning environment
    pl_env = get_3dof_planning_environment()
    if dof == 7:
        pl_env = get_7dof_planning_environment()

    # File paths
    hard_requests_file = f"../data/pl_req/{dof}dof/{dof}dof_hard_requests.json"
    easy_requests_file = f"../data/pl_req/{dof}dof/{dof}dof_easy_requests.json"
    graph_file_base = f"../data/graphs/{dof}dof/{dof}dof_{n_nodes}_graph_" + "{}.grahml"
    hard_rrt_solution_file = f"../data/pl_req/{dof}dof/rrt_solution/{dof}dof_hard_rrt_solution.json"
    easy_rrt_solution_file = f"../data/pl_req/{dof}dof/rrt_solution/{dof}dof_easy_rrt_solution.json"

    if not os.path.isdir(f"../data/pl_req/{dof}dof"):
        os.makedirs(f"../data/pl_req/{dof}dof")

    if not os.path.isdir(f"../data/graphs/{dof}dof"):
        os.makedirs(f"../data/graphs/{dof}dof")

    if not os.path.isdir(f"../data/pl_req/{dof}dof/rrt_solution"):
        os.makedirs(f"../data/pl_req/{dof}dof/rrt_solution")

    print("Generating graphs...")
    # Generate graphs
    for i in range(n_graphs):
        graph_file = graph_file_base.format(i)
        generate_graph(pl_env, n_nodes=n_nodes, graph_file=graph_file)

    pl_env.generate_graph_halton(n_nodes=n_nodes)

    print("Generating planning requests...")
    # Generate planning requests
    requests_config = {"n_data": 1000,
                       "hard_requests_file": hard_requests_file,
                       "easy_requests_file": easy_requests_file,
                       "hard_rrt_solution_file": hard_rrt_solution_file,
                       "easy_rrt_solution_file": easy_rrt_solution_file}
    generate_planning_requests(requests_config, pl_env=pl_env)

    print("Generating plan B...")
    for i in range(n_graphs):
        plan_b_file = f"../data/planB/{dof}dof/{dof}dof_graph_{i}_easy_requests_plan_b.json"
        graph_file = graph_file_base.format(i)
        if not os.path.isdir(f"../data/planB/{dof}dof"):
            os.makedirs(f"../data/planB/{dof}dof")

        config = {"graph_file": graph_file,
                  "requests_file": easy_requests_file,
                  "processed_plan_b_file": plan_b_file}
        record_blocked_position_plan_b_mapping(config, pl_env)
