import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.sparse as sp

from data_process import process_data
from dataloader import Hypergraph
from model_NHP import NHP
from test import test_model
from train import train

# set parameters
datasets = ['iAF1260b', 'iJO1366', 'uspto']
use_candidate = True
hidden_size = 512
g_functions = ['mean', 'maxmin']
split = 'train'
batch_size = 64
epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for g_func in g_functions:
    print(f"=============== {g_func} ===============")
    for dataset in datasets:
        auc_list = []
        recall_list = []
        for i in range(10):
            # preprocess data, randomly split train and test data
            process_data(dataset, use_candidate)

            # initialize model
            feature_size = sp.load_npz(f"datasets/{dataset}/features.npz").shape[1]
            model = NHP(feature_size, hidden_size, g_func)

            # load the dataset
            train_hypergraph = Hypergraph(dataset, split, batch_size)
            test_hypergraph = Hypergraph(dataset, "test", batch_size)
            train_data = DataLoader(train_hypergraph, batch_size=1, shuffle=False)
            test_data = DataLoader(test_hypergraph, batch_size=1, shuffle=False)

            # train NHP model
            trained_model = train(model, train_data, epochs, device, learning_rate)

            # test trained model on metrics
            auc, recall_at_k = test_model(trained_model, test_data)
            auc_list.append(auc)
            recall_list.append(recall_at_k)

        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)

        print(f"Dataset: {dataset}")
        print(f"AUC: {mean_auc:.2f} ± {std_auc:.2f}")
        print(f"Recall@k: {mean_recall:.2f} ± {std_recall:.2f}")
        if dataset != 'uspto':
            print("-------------------------------------")
