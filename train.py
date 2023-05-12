import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from pyg_dataset import *
from torch_geometric.loader import DataLoader
from model import GCN, GCNwLabels
import os

from torch.nn import BCELoss, CrossEntropyLoss, NLLLoss
import time


def train(save=False, dataset_type="global"):
    # Load data
    cwd = os.getcwd()
    if dataset_type == "local":
        name = "removal_partial_labelled_local"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalPartialLabelledDatasetLocal(root=data_root, hop=3, n_files=50, dataset_name=name)
    else:
        name = "removal_global"
        # name = "removal_partial_labelled_local"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalDatasetGlobal(root=data_root, dataset_name=name)

    dataset = dataset.shuffle()
    print(dataset.len())
    number_test_examples = int(dataset.len()*0.2)
    train_dataset = dataset[number_test_examples:]
    test_dataset = dataset[:number_test_examples]

    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Define the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNwLabels()
    # model_path = os.path.join(cwd, "../data/models/bets-gnn_node_classifier_1669035664040771787.pt")
    # model.load_state_dict(torch.load(model_path))

    model.double()
    model = model.to(device)
    criterion = NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = 1000
    train_loss = []
    best_acc = 0
    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0
        train_correct = 0
        train_data_cnt = 0
        for data in train_data_loader:
            cuda_data = data.to(device)
            optimizer.zero_grad()
            out = model(cuda_data)
            if dataset_type == "local":
                y = torch.argmax(cuda_data.y[cuda_data.train_mask], dim=1)
                loss = criterion(out[cuda_data.train_mask], y)
            else:
                y = torch.argmax(cuda_data.y, dim=1)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            if epoch % 20 == 0:
                pred = torch.argmax(out, dim=1)
                y = torch.argmax(cuda_data.y, dim=1)
                if dataset_type == "local":
                    train_correct += (pred[cuda_data.test_mask] == y[cuda_data.test_mask]).sum()
                    train_data_cnt += pred[cuda_data.test_mask].shape[0]
                else:
                    train_correct += (pred == y).sum()
                    train_data_cnt += pred.shape[0]

        train_loss.append(t_loss)

        if epoch % 20 == 0:
            print("Running evaluation for test datas")
            model.eval()
            correct = 0
            data_cnt = 0
            for data in test_data_loader:
                with torch.no_grad():
                    cuda_data = data.to(device)
                    cuda_data.x = cuda_data.x
                    out = model(cuda_data)
                    pred = torch.argmax(out, dim=1)
                    y = torch.argmax(cuda_data.y, dim=1)
                    if dataset_type == "local":
                        correct += (pred[cuda_data.test_mask] == y[cuda_data.test_mask]).sum()
                        data_cnt += pred[cuda_data.test_mask].shape[0]
                    else:
                        correct += (pred == y).sum()
                        data_cnt += pred.shape[0]

            acc = correct/data_cnt
            train_acc = train_correct/train_data_cnt
            best_acc = best_acc if best_acc > acc else acc
            print(f"Running epoch {epoch}/{epochs} - Accuracy: train {train_acc:.4f} test {acc:.4f} - Loss {t_loss:.4f}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if (epoch % 250 == 0 or acc >= best_acc) and epoch != 0:
                if save is True:
                    train_acc = int(train_acc*100)
                    acc = int(acc*100)
                    model_name = f"{dataset_type}_gnn_node_classifier_{epoch}_acc_{train_acc}_{acc}.pt"
                    path = os.path.join(os.getcwd(), "../data/models/{}".format(model_name))
                    torch.save(model.state_dict(), path)


    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()


if __name__ == "__main__":
    train(save=True, dataset_type="local")



