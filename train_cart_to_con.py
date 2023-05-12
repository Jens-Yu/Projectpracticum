import torch.utils.data.dataset

from utilities import *
from planning import Planning, get_2dof_planning_environment
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class RandomOccMap(Dataset):
    def __init__(self, request_file):
        self.requests = load_planning_requests(request_file)[:1000]
        self.data = self.prepare_data()

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        jnt = self.data[0][idx]
        occ = self.data[1][idx]
        occ_with_robot = self.data[2][idx]
        return jnt, occ, occ_with_robot

    def prepare_data(self):
        pl_env = get_2dof_planning_environment()
        occ_numpy = [pl_env.get_occupancy_map(req) for req in self.requests]
        occ_with_robot = [pl_env.get_occupancy_map_with_robot(req, req.start) for req in self.requests]
        occ_tensor = [torch.from_numpy(occ) for occ in occ_numpy]
        occ_tensor_with_robot = [torch.from_numpy(occ) for occ in occ_with_robot]
        jnt_tensor = [torch.from_numpy(req.start) for req in self.requests]
        return jnt_tensor, occ_tensor, occ_tensor_with_robot


def prepare_data(config, pl_env: Planning, data_type="train"):
    # Get configuration
    requests_file = config["requests_file"]
    number_of_static_obstacles = config["number_of_static_obstacles"]
    number_of_nodes = config["number_of_nodes"]
    number_of_requests = config["number_of_requests"]
    node_frequency_file = config["node_frequency_file"]

    # Get planning requests
    planning_requests = load_planning_requests(requests_file)
    end_index_request = len(planning_requests) - 1
    obs_planning_requests = planning_requests[-number_of_static_obstacles:]
    obs_planning_requests.reverse()

    # Prepare data
    iterable = range(number_of_static_obstacles-1) \
        if data_type == "train" else [number_of_static_obstacles - 1]
    data = {i: {} for i in iterable}
    print(iterable)
    print(len(obs_planning_requests))

    # Prepare occupancy map
    for i in iterable:
        node_frequency_file_i = node_frequency_file + f"_{end_index_request-i}.json"
        node_frequency = read_from_json(node_frequency_file_i)
        data[i]["occupancy_map"] = pl_env.get_occupancy_map(obs_planning_requests[i])
        data[i]["node_state"] = [node["state"] for _, node in node_frequency.items()]
        data[i]["counts"] = [node["counts"] for _, node in node_frequency.items()]
        # Normalize the node frequency
        mm = np.max(data[i]["counts"])
        data[i]["counts"] = data[i]["counts"] / mm
        ss = np.std(data[i]["counts"])
        # data[i]["counts"] = data[i]["counts"] / ss
        data[i]["max"] = mm
        data[i]["std"] = ss

    return data


def random_data(data, batch_size=1, data_type="train"):
    np.random.seed()
    n_static_obstacles = len(data)
    n_nodes = len(data[0]["node_state"])

    data_list = []
    # Select a static obstacle
    for i in range(batch_size):
        i_obs = np.random.randint(0, n_static_obstacles-1, batch_size)
        i_node = np.random.randint(0, n_nodes-1, batch_size)
        ii_node = i_node[i]
        ii_obs = i_obs[i] if data_type == "train" else list(data.keys())[0]

        # Get the node state and occupancy map
        x_node = data[ii_obs]["node_state"][ii_node]
        x_occ_map = data[ii_obs]["occupancy_map"].flatten()

        # Concatenate the node state and occupancy map
        # x_value = np.concatenate((x_node, x_occ_map), axis=0)

        # Get the node frequency
        y_value = data[ii_obs]["counts"][ii_node]

        # Add them into batch
        data_list.append((x_node, x_occ_map, y_value))

    x_np = np.array([data_list[i][0] for i in range(batch_size)])
    x_occ_np = np.array([data_list[i][1] for i in range(batch_size)])
    y_np = np.array([data_list[i][2] for i in range(batch_size)])

    return x_np, x_occ_np, y_np


def show_ranking(x):
    # Give ranking to the input according to its value
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    print("ranks", ranks)
    return ranks


def train_occ_map(config):
    from model import OccEncoder, OccDecoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get configuration
    requests_file = config["requests_file"]
    cae_encoder_model_file = config["cae_encoder_model_file"]
    cae_decoder_model_file = config["cae_decoder_model_file"]

    dataset = RandomOccMap(requests_file)

    # Split the dataset
    train_set, test_set = torch.utils.data.dataset.random_split(
        dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    dataloader = DataLoader(train_set.dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set.dataset, batch_size=1, shuffle=True)
    n_features = dataset[0][1].shape[0]

    # Get the model
    encoder = OccEncoder(n_features).double().to(device)
    decoder = OccDecoder(n_features).double().to(device)

    lr = 0.001
    weight_decay = 0.0001
    # Get the optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                 lr=lr, weight_decay=weight_decay)

    # Get the loss function
    loss_fn = torch.nn.MSELoss()

    epochs = 1000
    # Train the model
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        for jnt, x, y in dataloader:
            jnt = jnt.reshape(-1, 2).to(device)
            x = x.reshape(-1, 1, n_features, n_features).to(device)
            y = y.reshape(-1, 1, n_features, n_features).to(device)
            z = encoder(x, jnt)
            # z = torch.cat((z, jnt), dim=1)
            x_re = decoder(z)

            loss = loss_fn(y, x_re)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            encoder.eval()
            decoder.eval()
            for jnt, x, y in test_dataloader:
                jnt = jnt.reshape(-1, 2).to(device)
                x = x.reshape(-1, 1, n_features, n_features).to(device)
                y = y.reshape(-1, 1, n_features, n_features).to(device)
                z = encoder(x, jnt)
                # z = torch.cat((z, jnt), dim=1)
                x_re = decoder(z)

                _, ax = plt.subplots(1, 2, figsize=(10, 10))
                y_plot = y[0].squeeze().cpu().detach().numpy()
                x_re_plot = x_re[0].squeeze().cpu().detach().numpy()
                ax[0].imshow(y_plot)
                ax[1].imshow(x_re_plot)
                plt.show()
                # Save the model
                torch.save(encoder.state_dict(), cae_encoder_model_file)
                torch.save(decoder.state_dict(), cae_decoder_model_file)
                break
        print(f"Epoch {epoch}, loss {loss.item()}")

    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(y_plot)
    ax[1].imshow(x_re_plot)
    plt.show()


if __name__ == "__main__":
    frequency_config = {"requests_file": "../data/pl_req/easy_pl_req_250_nodes.json",
                        "number_of_nodes": 100,
                        "number_of_static_obstacles": 10,
                        "number_of_requests": 300,
                        "node_frequency_file": "../data/frequency/2dof/node_frequency",
                        "cae_encoder_model_file": "../data/models/cae_encoder.pt",
                        "cae_decoder_model_file": "../data/models/cae_decoder.pt"}

    if not os.path.isdir("../data/frequency/2dof"):
        os.makedirs("../data/frequency/2dof")

    train_occ_map(frequency_config)

    pl_env = get_2dof_planning_environment()
    data = prepare_data(frequency_config, pl_env)
    occ_map_size = np.shape(data[0]["occupancy_map"])[0]

    from model import C2CValueMap, OccEncoder, OccDecoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C2CValueMap(occ_map_size).double().to(device)
    encoder = OccEncoder(occ_map_size).double().to(device)
    encoder.load_state_dict(torch.load(frequency_config["cae_encoder_model_file"]))
    decoder = OccDecoder(occ_map_size).double().to(device)
    decoder.load_state_dict(torch.load(frequency_config["cae_decoder_model_file"]))

    # We only train the value map. The encoder is fixed
    lr = 0.0005
    weight_decay = 0.0001
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()),
    #                              lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    runs = 1000
    train_loss = []

    model.train()
    for i in range(runs):
        x, x_occ, y_gt = random_data(data, batch_size=32)
        x = torch.from_numpy(x).to(device)
        y_gt = torch.from_numpy(y_gt).to(device)
        x_occ = torch.from_numpy(x_occ).to(device)
        x_occ = x_occ.reshape(-1, 1, occ_map_size, occ_map_size)
        optimizer.zero_grad()
        y = model(x, encoder(x_occ))
        y = y.squeeze()

        loss = criterion(y_gt, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if i % 10 == 0:
            print(f"{i} - loss", loss.item())

        if i % 100 == 0:
            # z = encoder(x_occ)
            # x_re = decoder(z)
            #
            # _, ax = plt.subplots(1, 2, figsize=(10, 10))
            # x_plot = x_occ[0].squeeze().cpu().detach().numpy()
            # x_re_plot = x_re[0].squeeze().cpu().detach().numpy()
            # ax[0].imshow(x_plot)
            # ax[1].imshow(x_re_plot)
            # plt.show()
            print("y_gt", y_gt)
            print("y", y)

    y_gt = y_gt.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    # Save the model
    torch.save(model.state_dict(), f"../data/models/c2c_2dof_{runs}_runs_{loss.item()}.pt")

    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()