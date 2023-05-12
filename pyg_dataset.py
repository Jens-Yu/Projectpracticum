import numpy as np
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
import networkx as nx
from torch_geometric.data import Data
import torch
import os.path as osp
from utilities import *
import os
from model import GCN
from sklearn.preprocessing import StandardScaler


class LocalGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, train=True):
        self.processed_file_cnts = 0
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_dir)

    @property
    def raw_file_names(self):
        number_of_graph = 1
        if self.train:
            # TODO(Xi): Just two examples for good and useless nodes
            good_node_paths = [os.path.join(
                self.root, "nodes/good_nodes_{}.json".format(i)) for i in range(number_of_graph)]
            useless_node_paths = [os.path.join(
                self.root, "nodes/useless_nodes_{}.json".format(i)) for i in range(number_of_graph)]
        else:
            good_node_paths = [os.path.join(
                self.root, "nodes/good_nodes_{}.json".format(i)) for i in range(number_of_graph+1, number_of_graph+3)]
            useless_node_paths = [os.path.join(
                self.root, "nodes/useless_nodes_{}.json".format(i)) for i in range(number_of_graph+1, number_of_graph+3)]

        file_names = good_node_paths + useless_node_paths
        return file_names

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        return [osp.join(
            self.processed_dir, f'data_{idx}.pt') for idx in range(self.processed_file_cnts)]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            # Read important nodes or useless nodes
            json_dataset = read_from_json(raw_path)
            for json_data in json_dataset.values():
                # TODO(Xi): Just some dummy conditions to test the dataset

                gf = json_data["graph"]
                g = get_graph_from_grahml(gf, "state")

                # good or useless
                character = json_data["character"]
                samples = np.array(json_data["nodes"])
                for sample in samples:
                    n_nodes = g.number_of_nodes()
                    add_node(g, sample, remove_identical=(character == "useless"))

                    # Maybe we can use this directly
                    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
                    local_graph = filter_out_local_graph(global_graph=g, node_idx=n_nodes, hop=1)
                    adj = nx.to_scipy_sparse_matrix(local_graph).tocoo()
                    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
                    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
                    edge_index = torch.stack([row, col], dim=0)

                    # degrees = np.array(list(dict(G.degree()).values()))
                    # scale = StandardScaler()
                    # normalized_degrees = scale.fit_transform(degrees.reshape(-1, 1))

                    jnt_coordinates = np.array([], dtype=np.float64)

                    xyz = None
                    for node in local_graph.nodes:
                        coordinates = local_graph.nodes[node]['state']
                        xyz = np.array(coordinates, dtype=np.float64)
                        jnt_coordinates = np.append(jnt_coordinates, xyz)

                    jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])
                    labels = np.zeros(local_graph.number_of_nodes(), dtype=np.float64)
                    labels[-1] = 1.0 if character == "good" else 0.0
                    labels = torch.from_numpy(labels)

                    # Define the train mask only for the important/useless node
                    train_mask = torch.zeros(labels.size(0), dtype=torch.bool)
                    train_mask[-1] = True

                    test_mask = torch.zeros(labels.size(0), dtype=torch.bool)
                    test_mask[-1] = True

                    # Node labels
                    data = Data(x=torch.from_numpy(jnt_coordinates).double(),
                                edge_index=edge_index,
                                y=labels,
                                train_mask=train_mask,
                                test_mask=test_mask)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if self.train:
                        torch.save(data, osp.join(self.processed_dir, f'train_data_{idx}.pt'))
                    else:
                        torch.save(data, osp.join(self.processed_dir, f'test_data_{idx}.pt'))

                    self.processed_file_cnts += 1
                    idx += 1

    def len(self):
        return self.processed_file_cnts

    def get(self, idx):
        if self.train:
            data = torch.load(osp.join(self.processed_dir, f'train_data_{idx}.pt'))
        else:
            data = torch.load(osp.join(self.processed_dir, f'test_data_{idx}.pt'))
        return data


class LocalGraphInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, train=True):
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        number_of_graph = 1
        if self.train:
            # TODO(Xi): Just two examples for good and useless nodes
            good_node_paths = [os.path.join(
                self.root, "nodes/good_nodes_{}.json".format(i)) for i in range(number_of_graph)]
            useless_node_paths = [os.path.join(
                self.root, "nodes/useless_nodes_{}.json".format(i)) for i in range(number_of_graph)]
        else:
            good_node_paths = [os.path.join(
                self.root, "nodes/good_nodes_{}.json".format(i)) for i in range(number_of_graph+1, number_of_graph+60)]
            useless_node_paths = [os.path.join(
                self.root, "nodes/useless_nodes_{}.json".format(i)) for i in range(number_of_graph+1, number_of_graph+60)]

        file_names = good_node_paths + useless_node_paths
        return file_names

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        file_name = "in_memory_data_train.pt" if self.train else "in_memory_data_test.pt"
        return [osp.join(self.processed_dir, file_name)]

    def download(self):
        pass

    def process(self):
        idx = 0
        data_list = []
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            # Read important nodes or useless nodes
            json_dataset = read_from_json(raw_path)
            for json_data in json_dataset.values():
                # TODO(Xi): Just some dummy conditions to test the dataset

                gf = json_data["graph"]
                g = get_graph_from_grahml(gf, "state")

                # good or useless
                character = json_data["character"]
                samples = np.array(json_data["nodes"])
                for sample in samples:
                    n_nodes = g.number_of_nodes()
                    add_node(g, sample, remove_identical=(character == "useless"))
                    diff = g.number_of_nodes() - n_nodes

                    # Maybe we can use this directly
                    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
                    local_graph = filter_out_local_graph(global_graph=g, node_idx=n_nodes, hop=1)
                    adj = nx.to_scipy_sparse_matrix(local_graph).tocoo()
                    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
                    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
                    edge_index = torch.stack([row, col], dim=0)

                    # degrees = np.array(list(dict(G.degree()).values()))
                    # scale = StandardScaler()
                    # normalized_degrees = scale.fit_transform(degrees.reshape(-1, 1))

                    jnt_coordinates = np.array([], dtype=np.float64)

                    xyz = None
                    for node in local_graph.nodes:
                        coordinates = local_graph.nodes[node]['state']
                        xyz = np.array(coordinates, dtype=np.float64)
                        jnt_coordinates = np.append(jnt_coordinates, xyz)

                    jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])
                    labels = np.zeros(local_graph.number_of_nodes(), dtype=np.float64)
                    labels[-1] = 1.0 if character == "good" else 0.0
                    labels = torch.from_numpy(labels)

                    # Define the train mask only for the important/useless node
                    train_mask = torch.zeros(labels.size(0), dtype=torch.bool)
                    train_mask[-1] = True

                    test_mask = torch.zeros(labels.size(0), dtype=torch.bool)
                    test_mask[-1] = True

                    # Node labels
                    data = Data(x=torch.from_numpy(jnt_coordinates).double(),
                                edge_index=edge_index,
                                y=labels,
                                train_mask=train_mask,
                                test_mask=test_mask)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class FpFnDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        self.data_path = os.path.join(root, "processed/fp_fn_dataset.pt")
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        file_name = "fp_fn_dataset.pt"
        return [osp.join(self.processed_dir, file_name)]

    def download(self):
        pass

    def process(self):
        cwd = os.getcwd()
        data_root = os.path.join(cwd, "../data")

        test_dataset = LocalGraphInMemoryDataset(root=data_root, train=True)
        test_data_loader = DataLoader(test_dataset, shuffle=True)

        # Define the model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN()
        model_path = os.path.join(cwd, "../data/models/gnn_node_classifier_1669124831051338436.pt")
        model.load_state_dict(torch.load(model_path))

        model.double()
        model = model.to(device)

        model.eval()
        data_list = []

        for data in test_data_loader:
            with torch.no_grad():
                cuda_data = data.to(device)
                out = model(cuda_data)
                pred = torch.round(out)
                if pred[cuda_data.test_mask] != cuda_data.y[cuda_data.test_mask]:
                    new_data = Data(x=data.x.double()/torch.pi,
                                    edge_index=data.edge_index,
                                    y=data.y,
                                    train_mask=data.train_mask,
                                    test_mask=data.test_mask)
                    data_list.append(new_data)

        len_data = len(data_list)
        cnt = 0
        for data in test_data_loader:
            with torch.no_grad():
                cuda_data = data.to(device)
                out = model(cuda_data)
                pred = torch.round(out)
                if pred[cuda_data.test_mask] == cuda_data.y[cuda_data.test_mask]:
                    new_data = Data(x=data.x.double()/torch.pi,
                                    edge_index=data.edge_index,
                                    y=data.y,
                                    train_mask=data.train_mask,
                                    test_mask=data.test_mask)
                    data_list.append(new_data)
                    cnt += 1
            if cnt > len_data:
                break

        d, slices = self.collate(data_list)
        torch.save((d, slices), self.processed_paths[0])


class RemovalDatasetGlobal(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, training=True):
        self.data_path = os.path.join(root, "processed/" + dataset_name)
        self.training = training
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names_basis = \
            "/home/huang/ml_workspace/mp2d/data/compare/graph_250_nodes_{}_after_selected_removal.json"
        if self.training:
            return [file_names_basis.format(i) for i in range(51)]
        else:
            return [file_names_basis.format(51)]

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        file_name = "removal_dataset.pt"
        return [osp.join(self.processed_dir, file_name)]

    def download(self):
        pass

    def process(self):
        data_list = []
        # Load graph
        for raw_file in self.raw_file_names:
            # File format
            # {
            #     "0": {
            #         "nodes": [
            #             -2.999,
            #             0.637
            #         ],
            #         "graph": "/home/huang/ml_workspace/mp2d/scripts/../data/graphs/graph_250_nodes_99.graphml",
            #         "index": 0,
            #         "character": "important"
            #     },
            # }
            json_dict = read_from_json(raw_file)
            graph_file = json_dict["0"]["graph"]
            g = get_graph_from_grahml(graph_file, "state")
            json_values = list(json_dict.values())
            labels = [0 if v["character"] == "useless" else 1 for v in json_values]

            adj = nx.to_scipy_sparse_matrix(g).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)

            degrees = np.array(list(dict(g.degree()).values()))
            scale = StandardScaler()
            normalized_degrees = scale.fit_transform(degrees.reshape(-1, 1))

            jnt_coordinates = np.array([], dtype=np.float64)

            xyz = None
            for i, node in enumerate(g.nodes):
                xyz = g.nodes[node]["state"]/np.pi
                xyz = xyz.tolist()
                d = normalized_degrees[i]
                xyz.append(d)
                xyz = np.array(xyz, dtype=np.float64)
                jnt_coordinates = np.append(jnt_coordinates, xyz)

            jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])
            labels = np.array(labels, dtype=np.float64)
            labels = torch.from_numpy(labels)

            # Define the train mask only for the important/useless node
            # Node labels
            new_data = Data(x=torch.from_numpy(jnt_coordinates).double(),
                            edge_index=edge_index,
                            y=labels)
            # Not sure if we should split the data here.
            # new_data = RandomNodeSplit(num_val=0.1, num_test=0.2)(new_data)
            data_list.append(new_data)
        d, slices = self.collate(data_list)
        torch.save((d, slices), self.processed_paths[0])


class RemovalDatasetLocal(InMemoryDataset):
    def __init__(self, root, dataset_name, hop=1, transform=None, training=True):
        root = root + "_{}_hop".format(hop)
        self.data_path = os.path.join(root, "processed/" + dataset_name)
        self.hop = hop
        self.training = training
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names_basis = \
            "/home/huang/ml_workspace/mp2d/data/compare/graph_250_nodes_{}_after_selected_removal.json"

        if self.training:
            return [file_names_basis.format(i) for i in range(51)]
        else:
            return [file_names_basis.format(51)]

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        file_name = "removal_dataset.pt"
        return [osp.join(self.processed_dir, file_name)]

    def download(self):
        pass

    def process(self):
        data_list = []
        # Load graph
        for raw_file in self.raw_file_names:
            # File format
            # {
            #     "0": {
            #         "nodes": [
            #             -2.999,
            #             0.637
            #         ],
            #         "graph": "/home/huang/ml_workspace/mp2d/scripts/../data/graphs/graph_250_nodes_99.graphml",
            #         "index": 0,
            #         "character": "important"
            #     },
            # }
            json_dict = read_from_json(raw_file)
            graph_file = json_dict["0"]["graph"]
            g = get_graph_from_grahml(graph_file, "state")
            json_values = list(json_dict.values())
            labels = [0 if v["character"] == "useless" else 1 for v in json_values]
            samples = nx.get_node_attributes(g, "state")

            degrees = np.array(list(dict(g.degree()).values()))
            scale = StandardScaler()
            normalized_degrees = scale.fit_transform(degrees.reshape(-1, 1))

            for i, sample in samples.items():
                # Maybe we can use this directly
                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
                local_graph = filter_out_local_graph(global_graph=g, node_idx=i, hop=self.hop)
                adj = nx.to_scipy_sparse_matrix(local_graph).tocoo()
                row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
                col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
                edge_index = torch.stack([row, col], dim=0)

                jnt_coordinates = np.array([], dtype=np.float64)

                xyz = None
                center_node_idx = None
                cnt = 0
                for node in local_graph.nodes:
                    xyz = g.nodes[node]["state"]
                    if np.linalg.norm(xyz-sample) < 1e-6:
                        assert (center_node_idx is None)
                        center_node_idx = cnt

                    xyz = (xyz/np.pi).tolist()  # Normalize
                    d = normalized_degrees[node]  # Degree of the global graph
                    xyz.append(d)
                    xyz = np.array(xyz, dtype=np.float64)
                    jnt_coordinates = np.append(jnt_coordinates, xyz)
                    cnt += 1
                # Find node index

                jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])
                local_labels = np.zeros(local_graph.number_of_nodes(), dtype=np.float64)
                local_labels[center_node_idx] = labels[i]
                local_labels = torch.from_numpy(local_labels)

                # Define the train mask only for the important/useless node
                train_mask = torch.zeros(local_labels.size(0), dtype=torch.bool)
                train_mask[center_node_idx] = True

                test_mask = torch.zeros(local_labels.size(0), dtype=torch.bool)
                test_mask[center_node_idx] = True

                # print(i, jnt_coordinates[center_node_idx]*np.pi, sample)

                # Node labels
                data = Data(x=torch.from_numpy(jnt_coordinates).double(),
                            edge_index=edge_index,
                            y=local_labels,
                            train_mask=train_mask,
                            test_mask=test_mask)
                data_list.append(data)

        d, slices = self.collate(data_list)
        torch.save((d, slices), self.processed_paths[0])


class RemovalPartialLabelledDatasetLocal(InMemoryDataset):
    def __init__(self, root, dataset_name, hop=1, n_files=1, transform=None, training=True):
        root = root + "_{}_hop".format(hop)
        self.data_path = os.path.join(root, "processed/" + dataset_name)
        self.hop = hop
        self.training = training
        self.n_files = n_files
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names_basis = \
            "/home/huang/ml_workspace/mp2d/data/compare/graph_250_nodes_{}_after_selected_removal.json"

        if self.training:
            return [file_names_basis.format(i) for i in range(self.n_files)]
        else:
            return [file_names_basis.format(self.n_files)]

    @property
    def processed_file_names(self):
        # TODO(Xi): find out how does it work
        file_name = "removal_dataset.pt"
        return [osp.join(self.processed_dir, file_name)]

    def download(self):
        pass

    def process(self):
        data_list = []
        # Load graph
        for raw_file in self.raw_file_names:
            # File format
            # {
            #     "0": {
            #         "nodes": [
            #             -2.999,
            #             0.637
            #         ],
            #         "graph": "/home/huang/ml_workspace/mp2d/scripts/../data/graphs/graph_250_nodes_99.graphml",
            #         "index": 0,
            #         "character": "important"
            #     },
            # }
            json_dict = read_from_json(raw_file)
            graph_file = json_dict["0"]["graph"]
            g = get_graph_from_grahml(graph_file, "state")
            json_values = list(json_dict.values())
            # Label format: important - [1, 0], useless - [0, 1], unknown - [0, 0]

            node_usage_counts = [v["counts"] for v in json_values]
            node_usage_std = np.std(node_usage_counts)
            node_usage_mean = np.mean(node_usage_counts)
            normalized_node_usage = (node_usage_counts - node_usage_mean)/node_usage_std

            labels = [[0, 1] if v["character"] == "useless" else [1, 0]
                      for v in json_values]

            #filter_out_close_nodes(g, labels)

            # Assign_labels to the graph:
            for node in g.nodes:
                g.nodes[node]["label"] = labels[node]

            samples = nx.get_node_attributes(g, "state")

            degrees = np.array(list(dict(g.degree()).values()))
            scale = StandardScaler()
            normalized_degrees = scale.fit_transform(degrees.reshape(-1, 1))

            for i, sample in samples.items():
                # Maybe we can use this directly
                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
                local_graph = filter_out_local_graph(global_graph=g, node_idx=i, hop=self.hop, remove_close_nodes=True)

                adj = nx.to_scipy_sparse_matrix(local_graph).tocoo()
                np_row = adj.row.astype(np.int64)
                np_col = adj.col.astype(np.int64)
                row = torch.from_numpy(np_row).to(torch.long)
                col = torch.from_numpy(np_col).to(torch.long)
                edge_index = torch.stack([row, col], dim=0)

                # Prepare edge weights
                np_edge_weight = np.zeros_like(np_row, dtype=np.float64)
                node_mapping = list(local_graph.nodes)
                for ei in range(np_edge_weight.shape[0]):
                    u = node_mapping[np_row[ei]]
                    v = node_mapping[np_col[ei]]
                    diff = np.linalg.norm(local_graph.nodes[u]["state"] - local_graph.nodes[v]["state"])
                    np_edge_weight[ei] = diff

                jnt_coordinates = np.array([], dtype=np.float64)
                local_labels = np.array([], dtype=np.float64)

                xyz = None
                label = None
                center_node_idx = None
                cnt = 0
                unknown_label_ratio = 0.0
                for node in local_graph.nodes:
                    xyz = g.nodes[node]["state"]
                    if np.linalg.norm(xyz-sample) < 1e-6:
                        assert (center_node_idx is None)
                        center_node_idx = cnt

                    random_number = np.random.random()  # Output random number in [0, 1)
                    # TODO: Wrong! We can not use this count to find the label in the global graph
                    if center_node_idx == cnt or random_number < unknown_label_ratio:
                        label = [0, 0]
                    else:
                        label = local_graph.nodes[node]["label"]

                    xyz = (xyz/np.pi).tolist()  # Normalize
                    d = normalized_degrees[cnt].tolist()  # Degree of the global graph
                    node_usage = normalized_node_usage[node]
                    xyz = xyz + d + label + [node_usage]
                    # Add label to input
                    xyz = np.array(xyz, dtype=np.float64)
                    jnt_coordinates = np.append(jnt_coordinates, xyz)
                    local_labels = np.append(
                        local_labels, np.array(local_graph.nodes[node]["label"], dtype=np.float64))
                    cnt += 1

                # We need to select assign labels to nodes in the surrounding
                jnt_coordinates = jnt_coordinates.reshape(-1, xyz.shape[0])
                local_labels = local_labels.reshape(jnt_coordinates.shape[0], -1)  # Not sure about this
                # Define the train mask only for the important/useless node
                train_mask = torch.zeros(local_labels.shape[0], dtype=torch.bool)
                train_mask[center_node_idx] = True

                test_mask = torch.zeros(local_labels.shape[0], dtype=torch.bool)
                test_mask[center_node_idx] = True

                # print(i, jnt_coordinates[center_node_idx]*np.pi, sample)

                # Node labels
                data = Data(x=torch.from_numpy(jnt_coordinates).double(),
                            edge_index=edge_index,
                            edge_attr=torch.from_numpy(np_edge_weight).double(),
                            y=torch.from_numpy(local_labels).double(),
                            train_mask=train_mask,
                            test_mask=test_mask)
                data_list.append(data)

        d, slices = self.collate(data_list)
        torch.save((d, slices), self.processed_paths[0])


if __name__ == "__main__":
    cwd = os.getcwd()
    # local_graph_dataset_train = LocalGraphInMemoryDataset(root=data_root, train=True)
    # local_graph_dataset_test = LocalGraphInMemoryDataset(root=data_root, train=False)
    # # visualize_dataset_distributions(root=data_root)
    name = "removal_partial_labelled_local_ttttest"
    data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
    #dataset = RemovalDatasetGlobal(root=data_root, dataset_name=name)
    dataset = RemovalPartialLabelledDatasetLocal(root=data_root, hop=2, n_files=1, dataset_name=name)

    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # visualize_global_dataset_distributions(loader)

