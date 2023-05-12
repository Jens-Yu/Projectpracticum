import numpy
import numpy as np

from train_cart_to_con import prepare_data, random_data
from planning import *
from model import C2CValueMap, OccEncoder, OccDecoder
import matplotlib.pyplot as plt
from utilities import *


def get_all_data(data):
    np.random.seed()
    n_nodes = len(data["node_state"])

    data_list = []
    # Select a static obstacle
    for i in range(n_nodes):
        # Get the node state and occupancy map
        x_node = data["node_state"][i]
        x_occ_map = data["occupancy_map"].flatten()

        # Concatenate the node state and occupancy map
        # x_value = np.concatenate((x_node, x_occ_map), axis=0)

        # Get the node frequency
        y_value = data["counts"][i]

        # Add them into batch
        data_list.append((x_node, x_occ_map, y_value))

    x_np = np.array([data_list[i][0] for i in range(n_nodes)])
    x_occ_np = np.array([data_list[i][1] for i in range(n_nodes)])
    y_np = np.array([data_list[i][2] for i in range(n_nodes)])

    return x_np, x_occ_np, y_np


def test(config):
    pl_env = get_2dof_planning_environment()
    data = prepare_data(config, pl_env, "test")
    x, occ_map, y = get_all_data(list(data.values())[0])
    occ_map_size = np.shape(list(data.values())[0]["occupancy_map"])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C2CValueMap(feature_size=occ_map_size).double().to(device)
    model.load_state_dict(torch.load("../data/models/c2c_2dof_1000_runs_0.008367584872888169.pt"))
    encoder = OccEncoder(feature_size=occ_map_size).double().to(device)
    encoder.load_state_dict(torch.load(config["cae_encoder_model_file"]))
    model.eval()
    encoder.eval()

    # Prepare contour plot
    n_contour = 80
    xs = np.linspace(-np.pi, np.pi, n_contour)
    ys = np.linspace(-np.pi, np.pi, n_contour+1)

    nodes = [np.array([x, y]) for x in xs for y in ys]
    nodes_tensor = [torch.from_numpy(ns) for ns in nodes]
    occ_tensor = torch.from_numpy(occ_map).to(device)

    nodes_tensor = torch.stack(nodes_tensor).to(device)
    print(nodes_tensor.shape)

    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        occ_tensor = occ_tensor.view(-1, 1, occ_map_size, occ_map_size)
        occ_tensor = encoder(occ_tensor)
        y_pred = model(x, occ_tensor)
        y_pred = y_pred / torch.max(y_pred)
        y_pred = y_pred.detach().cpu().numpy()
        occ_tensor_stack = torch.stack([occ_tensor[0] for _ in range(n_contour*(n_contour + 1))])
        y_contour = model(nodes_tensor, occ_tensor_stack)
        y_contour = y_contour / torch.max(y_contour)
        y_contour = y_contour.view(n_contour, n_contour+1)
        y_contour = y_contour.detach().cpu().numpy()

    print(y_pred)
    print(y)

    y = y / numpy.max(y)
    x = x.detach().cpu().numpy()

    # Visualize the result
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    ax[0].scatter(x[:, 0], x[:, 1], c=y, cmap="jet")
    ax[0].set_title("Ground truth")
    ax[1].scatter(x[:, 0], x[:, 1], c=y_pred, cmap="jet")
    ax[1].set_title("Prediction")
    ax[2].contourf(xs, ys, np.transpose(y_contour), cmap="jet")
    ax[2].set_title("Contour")

    for i in range(3):
        set_plot_square(ax[i])
    plt.show()


if __name__ == "__main__":
    frequency_config = {"requests_file": "../data/pl_req/easy_pl_req_250_nodes.json",
                        "number_of_nodes": 100,
                        "number_of_static_obstacles": 12,
                        "number_of_requests": 300,
                        "node_frequency_file": "../data/frequency/2dof/node_frequency",
                        "cae_encoder_model_file": "../data/models/cae_encoder.pt",
                        "cae_decoder_model_file": "../data/models/cae_decoder.pt"
                        }

    test(frequency_config)
