import matplotlib.pyplot as plt
import numpy as np

from utilities import *
from generate_classification_dataset import *
from train_GMM import prepare_data, random_data
import torch
# from evaluation import benchmarking_two_roadmaps
from model import FEST
from test_GMM import save_hard_request_figures
from storage_place import process_data

# TODO(Xi) - We need to include all information in a function
req_examples = [2, 4, 7, 13, 19]
for req in req_examples:
    process_config = {"node_usage_file": f"../data/nodes/node_usages_halton_graph_with_obstacles_{req}.json",
                      "original_path_file": f"../data/compare/original_path_halton_graph_with_obstacles_{req}.json",
                      "processed_path_file": f"../data/compare/processed_path_halton_graph_with_obstacles_{req}.json",
                      "processed_plan_b_file": f"../data/compare/processed_pb_halton_graph_with_obstacles_{req}.json",
                      "training_config": "",
                      "graph_file": f"../data/graphs/halton_graph_with_obstacles_{req}.graphml",
                      "requests_file": "../data/pl_req/easy_pl_req_250_nodes.json"}

    # process_data(process_config)

pl_env = get_2dof_planning_environment()

planning_request = "../data/pl_req/easy_pl_req_250_nodes.json"
node_usage_cnts_file_basis = "../data/nodes/structured_graph_nodes_usage_{}.json"
graph_path = "../data/graphs/structure_graph.graphml"
g_obs = get_graph_from_grahml(graph_path, attr="state")
print(g_obs.number_of_edges())
graph_path = "../data/graphs/halton_graph_with_obstacles_2.graphml"
g_obs = get_graph_from_grahml(graph_path, attr="state")
print("weighted", g_obs.number_of_edges())

easy_requests = load_planning_requests(planning_request)


#visualize_graphs([g_obs], attr="state", show_edges=True)
#save_graph_to_file(g, graph_file=graph_path)


obstacles = [easy_requests[i].obstacles for i in req_examples]
file_names = [f'../data/graphs/halton_graph_with_obstacles_{i}.graphml' for i in req_examples]
# generate_graphs_with_obstacles(obstacles, n_nodes=150, file_names=file_names)


file_name_basis = "../data/compare/structured_graph_after_selected_removal.json"
# results_before_removing = {"success_sum": 4754,
#                        "checked_counts": 4637502,
#                        "path_length": 27180.793973936416}

file_name = "../data/compare/structured_graph_blocked_and_planB.json"
processed_file_name = "../data/compare/structured_graph_planB_data.json"

# for idx in req_examples:
#     gf = f'../data/graphs/halton_graph_with_obstacles_{idx}.graphml'
#     node_file_basis = '../data/nodes/node_usages_halton_graph_with_obstacles'
#     record_node_usage(process_config)

# record_original_path(node_usage_cnts_file_basis.format(0), file_name)
# process_original_path_data_file(file_name, processed_file_name)

# g = get_graph_from_grahml(graph_path, "state")

# benchmarking_two_roadmaps(g, g_m)

# record_blocked_position_and_plan_bs(node_usage_cnts_file_basis.format(0), file_name)

# get_hard_request_planning_data()

# save_hard_request_figures()

#################################
# map_config = {"k_gmm": 20,
#               "gmm_model_file": "../data/models/GMM_model_158.35598754882812.pt",
#               "fest_model_file": "../data/models/FEST_value_model.pt"}
#
# from storage_place import map_static_blocked_region_to_plan_b
# xs, ys, wmap = map_static_blocked_region_to_plan_b(map_config, pl_env, easy_requests[2])
# print(np.shape(wmap))
# plt.contourf(xs, ys, np.transpose(wmap))
# plt.show()
#################################

# Get frequency of different settings
from multiple_dof_data_collection import record_frequency_with_obstacles
frequency_config = {"requests_file": "../data/pl_req/easy_pl_req_250_nodes.json",
                    "number_of_nodes": 100,
                    "number_of_static_obstacles": 15,
                    "number_of_requests": 300,
                    "node_frequency_file": "../data/frequency/2dof/node_frequency"}

if not os.path.isdir("../data/frequency/2dof"):
    os.makedirs("../data/frequency/2dof")

pl_env = get_2dof_planning_environment()
record_frequency_with_obstacles(frequency_config, pl_env)
