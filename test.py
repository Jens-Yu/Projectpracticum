import torch
from torch_geometric.loader import DataLoader
import os
import time
from train import GCN

# Metrics for evaluation
from sklearn.metrics import *
from pyg_dataset import *


def test(model_path, plot_all_wrong_pred=False):
    # Load data
    cwd = os.getcwd()
    dataset_type = "local"
    if dataset_type == "local":
        name = "removal_local_test"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalDatasetLocal(root=data_root, hop=2, dataset_name=name, training=False)
    else:
        name = "removal_global"
        data_root = os.path.join(cwd, "../data/dataset/{}".format(name))
        dataset = RemovalDatasetGlobal(root=data_root, dataset_name=name)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


    # Define the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN()
    model_path = os.path.join(cwd, model_path)
    model.load_state_dict(torch.load(model_path))

    model.double()
    model = model.to(device)

    model.eval()
    y_pred_list = []
    out_list = []
    y_gt_list = []
    fp_list = []

    for data in data_loader:
        with torch.no_grad():
            cuda_data = data.to(device)
            cuda_data.x = cuda_data.x
            out = model(cuda_data)
            pred = torch.round(out)

            # Save the data for visualization
            out_list.append(out[cuda_data.test_mask].cpu().detach().numpy())
            y_pred_list.append(pred[cuda_data.test_mask].cpu().detach().numpy())
            y_gt_list.append(cuda_data.y[cuda_data.test_mask].cpu().detach().numpy())
            if pred[cuda_data.test_mask] != cuda_data.y[cuda_data.test_mask] and cuda_data.y[cuda_data.test_mask] == 1:
                fp_list.append(data.x[data.test_mask].cpu().detach().numpy())

    ytest_pred = [a.squeeze().tolist() for a in y_pred_list]
    ytest_gt = [a.squeeze().tolist() for a in y_gt_list]

    if plot_all_wrong_pred:
        len_data = len(ytest_gt)
        wrong_prediction = [out_list[i].item() for i in range(len_data) if ytest_gt[i] != ytest_pred[i]]
        gt_wrong_prediction = [ytest_gt[i] for i in range(len_data) if ytest_gt[i] != ytest_pred[i]]
        for i, pred in enumerate(wrong_prediction):
            print("Wrong prediction {:.4f} : ".format(pred), ", ground truth: ", gt_wrong_prediction[i])

    conf_matrix = confusion_matrix(ytest_gt, ytest_pred)
    print("Accuracy  (Tp + Tn)/All:\t" + str(accuracy_score(ytest_gt, ytest_pred)))
    print("Precision Tp/(Tp + Fp) :\t" + str(precision_score(ytest_gt, ytest_pred)))
    print("Recall    Tp/(Tp + Fn) :\t" + str(recall_score(ytest_gt, ytest_pred)))
    print("F1 Score               :\t" + str(f1_score(ytest_gt, ytest_pred)))

   # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
   # disp.plot()
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(20, 14))
    print(fp_list)
    fp_list = [a.squeeze().tolist() for a in fp_list]
    fp_x = [fp[0]*torch.pi for fp in fp_list]
    fp_y = [fp[1]*torch.pi for fp in fp_list]

    ax.scatter(fp_x, fp_y)
    sum_links_length = 3.5
    ax.set_xlim([-sum_links_length, sum_links_length])
    ax.set_ylim([-sum_links_length, sum_links_length])
    ax.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    model_path = "../data/models/global_gnn_node_classifier_500_acc_70_68.pt"
    test(model_path)


