from utilities import *

from torch_geometric.loader import DataLoader
from model import GCNGMM, GMM, FEST
import os

from torch.nn import GaussianNLLLoss
import time

import torch.distributions as D
from torch.autograd import Variable


def random_data(data, batch_size=1):
    np.random.seed()
    data_keys = list(data.keys())
    n_data = len(data_keys)
    random_int = np.random.randint(0, n_data-1, batch_size)

    x = [data[data_keys[i]]["node"] for i in random_int]  # A node in numpy array
    y = [data[data_keys[i]]["plan_b"] for i in random_int]  # A list of nodes

    x_np = np.array(x)
    return x_np, y[0]


def prepare_data(config, data_type="train"):
    data_file = config["plan_b_data"]
    graph_file = config["graph"]
    g = get_graph_from_grahml(graph_file, "state")
    n_nodes = g.number_of_nodes()
    node_states = nx.get_node_attributes(g, "state")
    data = read_from_json(data_file)

    prepared_data = {}

    np.random.seed(10086)
    n_training = int(n_nodes * 0.8)
    data_index = set(range(n_nodes))
    training_data_index = np.random.randint(0, high=n_nodes, size=n_training).tolist()
    training_data_index = set(training_data_index)
    test_data_index = data_index - training_data_index

    dataset = list(training_data_index) if data_type == "train" else list(test_data_index)

    for i in dataset:
        plan_b_node_index = data[str(i)]["planB_nodes"]
        if not plan_b_node_index:
            continue
        plan_b_node_states = [node_states[idx] for idx in plan_b_node_index]
        plan_b_node_states_np = np.array(plan_b_node_states)
        prepared_data[i] = {"node": node_states[i], "plan_b": plan_b_node_states_np}

    return prepared_data


def train(save=False):
    # Define the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_gmm = 20
    model = GMM(k_gmm)
    # model_path = os.path.join(cwd, "../data/models/bets-gnn_node_classifier_1669035664040771787.pt")
    # model.load_state_dict(torch.load(model_path))

    model.double()
    model = model.to(device)
    criterion = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    epochs = 80
    train_loss = []
    best_acc = 0

    d = prepare_data(config)
    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0
        n_data = 64

        optimizer.zero_grad()
        loss = Variable(torch.tensor(.0)).to(device)
        for i in range(n_data):
            x, y = random_data(d)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            mean, log_std, co_eff, log_w = model(Variable(x))

            mean = mean.reshape(k_gmm, -1)
            sigma = log_std.reshape(k_gmm, -1).exp()

            cov_mat_list = form_cov_matrix(sigma, co_eff, k_gmm)
            dist_list = [D.MultivariateNormal(mean[ik], scale_tril=cov_mat_list[ik])
                         for ik in range(k_gmm)]

            # Compute log probabilities
            y = Variable(y)

            log_w = log_w.squeeze()

            exp_log_py_list = [torch.exp(dist_list[id].log_prob(y) + log_w[id])
                               for id in range(k_gmm)]

            exp_log_py = torch.stack(exp_log_py_list)

            log_py = torch.log(torch.sum(exp_log_py, dim=0))

            # print(f" diag {log_p_y[0]} -- mn {log_p_y_mn[0]}")
            loss += torch.mean(-log_py)

        loss.backward()
        optimizer.step()
        t_loss += loss.item()

        train_loss.append(t_loss)

    model_name = config["model_file_basis"].format(t_loss)
    torch.save(model.state_dict(), model_name)

    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()


def train_value_estimator(config):
    """
    Load a GMM model.
    Train value estimation models for the data with and without obstacles.
    Use the trained model for data with obstacles and combine it with GMM ->
        Visualize the output of GMM given the nodes -> Where are important for these nodes
    (Did not consider the existence of other nodes)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = get_graph_from_grahml(config["graph"], "state")
    original_path_data = read_from_json(config["original_path"])
    node_usages_without_obstacles = original_path_data["node_usages_without_obstacles"]
    node_usages_with_obstacles = original_path_data["node_usages_with_obstacles"]

    original_node_usages_w = np.array(list(node_usages_with_obstacles.values()))
    original_node_usages_wo = np.array(list(node_usages_without_obstacles.values()))

    scaling_factor = max(np.max(original_node_usages_w), np.max(original_node_usages_wo))

    original_node_usages_wo = original_node_usages_wo / scaling_factor
    original_node_usages_w = original_node_usages_w / scaling_factor

    epoch = 150
    from model import C2CValueMap
    model_w = FEST().double()
    model_wo = FEST().double()
    model_w.to(device)
    model_wo.to(device)

    model_w.train()
    model_wo.train()

    node_states = [torch.from_numpy(g.nodes[i]["state"]) for i in g.nodes]
    node_state_tensor = torch.stack(node_states).to(device)
    node_state_tensor = node_state_tensor.reshape(-1, 2)

    original_node_usages_wo_tensor = torch.from_numpy(original_node_usages_wo).to(device)
    original_node_usages_w_tensor = torch.from_numpy(original_node_usages_w).to(device)

    original_node_usages_wo_tensor = original_node_usages_wo_tensor.reshape(-1, 1)
    original_node_usages_w_tensor = original_node_usages_w_tensor.reshape(-1, 1)

    optim_wo = torch.optim.Adam(model_wo.parameters(), lr=0.005, weight_decay=5e-4)
    optim_w = torch.optim.Adam(model_w.parameters(), lr=0.005, weight_decay=5e-4)

    criterion = torch.nn.MSELoss()
    train_loss = []

    for i in range(epoch):
        # Train the model for data with obstacles
        optim_w.zero_grad()
        y_w = model_w(node_state_tensor)
        print(y_w)
        loss_w = criterion(original_node_usages_w_tensor, y_w)
        loss_w.backward()
        optim_w.step()
        train_loss.append(loss_w.item())

        # Train the model for data without obstacles
        optim_wo.zero_grad()
        y_wo = model_wo(node_state_tensor)
        loss_wo = criterion(original_node_usages_wo_tensor, y_wo)
        loss_wo.backward()
        optim_wo.step()

    model_name_w = "FEST_value_model.pt"
    model_name_wo = "FEST_value_model_wo.pt"

    path_w = os.path.join(os.getcwd(), "../data/models/{}".format(model_name_w))
    path_wo = os.path.join(os.getcwd(), "../data/models/{}".format(model_name_wo))

    # torch.save(model_w.state_dict(), path_w)
    # torch.save(model_wo.state_dict(), path_wo)

    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()


if __name__ == "__main__":
    gmm_config = {"graph": "../data/graphs/halton_graph_with_obstacles_2.graphml",
                  "plan_b_data": "../data/compare/processed_path_halton_graph_with_obstacles_2.json",
                  "model_file_basis": "../data/models/GMM_halton_graph_with_obstacles_2_{}.pt"}

    #train(gmm_config)

    fest_config = {"graph": "../data/graphs/halton_graph_with_obstacles_2.graphml",
                   "w_model": "../data/models/FEST_halton_graph_with_obstacles.pt",
                   "wo_model": "../data/models/FEST_halton_graph_without_obstacles_.pt",
                   "original_path": "../data/compare/original_path_halton_graph_with_obstacles_2.json"}

    train_value_estimator(fest_config)



