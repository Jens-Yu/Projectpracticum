# Model collection for learning

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TopKPooling, MLP

import torch.nn.functional as F
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = SAGEConv(32, 128)
        self.conv4 = SAGEConv(128, 32)
        self.FCN = MLP(in_channels=32, hidden_channels=16, out_channels=1, num_layers=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.FCN(x)
        return torch.squeeze(torch.sigmoid(x))


class GCNGMM(torch.nn.Module):
    def __init__(self, k_gmm=10):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 64)
        # self.conv4 = SAGEConv(128, 32)
        # In this case, we should care about the
        self.top_k_pooling = TopKPooling(64, ratio=k_gmm)
        self.fc_mean = Linear(64, 2)
        self.fc_logstd = Linear(64, 2)
        self.fc_w = Linear(64, 1)
        self.k_gmm = k_gmm

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Pooling
        print("Before pooling: ", x.shape)
        x = self.top_k_pooling(x, edge_index)
        x = x[0]
        print("After pooling: ", x.shape)

        # Get GMM parameters
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        w = F.softmax(self.fc_w(x), dim=1)

        return mean, log_std, w


class GMM(torch.nn.Module):
    def __init__(self, k_gmm=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, k_gmm*6)  # Maybe also correlations
        self.k_gmm = k_gmm

    def forward(self, data):
        # Layer 1
        x = self.fc1(data)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Get GMM parameters
        x = self.fc4(x)
        mean = x[:, :2*self.k_gmm]
        log_std = x[:, 2*self.k_gmm:4*self.k_gmm]
        co_eff = torch.tanh(x[:, 4*self.k_gmm:5*self.k_gmm])
        log_w = F.log_softmax(x[:, 5*self.k_gmm:], dim=1)

        return mean, log_std, co_eff, log_w


class FEST(torch.nn.Module):
    """
    Frequency Estimator
    """
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 128)
        self.fc4 = torch.nn.Linear(128, 1)  # Maybe also correlations

    def forward(self, data):
        # Layer 1
        x = self.fc1(data)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # Get GMM parameters
        x = self.fc4(x)
        x = F.relu(x)
        return x


class OccEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.dof = 2
        self.channels = [8, 16, 32]
        # Should we use sequence of convolutions?
        self.fc_input_size = int(feature_size ** 2) * self.channels[-1] + 2
        self.feature_size = feature_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.channels[0], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, return_indices=True),
            torch.nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.channels[1]),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1),
            torch.nn.ReLU())

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.feature_size),
            torch.nn.ReLU())

    def forward(self, occ, x):
        occ = self.conv(occ)
        occ = occ.view(-1, self.fc_input_size-2)
        x = torch.cat((occ, x), dim=1)
        x = self.fc(x)
        return x


class OccDecoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.channels = [8, 16, 32]
        self.fc_input_size = int(feature_size ** 2) * self.channels[-1]
        self.feature_size = feature_size
        self.indices = None
        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.fc_input_size),
            torch.nn.ReLU())

        self.decoder_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.channels[2], self.channels[1], kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.channels[1]),
            torch.nn.ReLU(),
            #torch.nn.MaxUnpool2d(2),
            torch.nn.ConvTranspose2d(self.channels[1], self.channels[0], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxUnpool2d(2),
            torch.nn.ConvTranspose2d(self.channels[0], 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = self.decoder_fc(x)
        x = x.view(-1, self.channels[2], int(self.feature_size), int(self.feature_size))
        x = self.decoder_conv(x)
        return x


class C2CValueMap(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.dof = 2

        self.fc_occ = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU())

        self.fc_x = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU())

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32*2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.ReLU())

    def forward(self, x, occ_features):
        # Concatenate
        occ = self.fc_occ(occ_features)
        x = self.fc_x(x)
        x = torch.cat([x, occ], dim=1)
        y = self.fc(x)
        return y








