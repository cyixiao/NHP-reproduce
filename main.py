import torch
from torch.utils.data import DataLoader
import scipy.sparse as sp

from dataloader import Hypergraph
from model_NHP import NHP
from test import test_model
from train import train

dataset = 'iJO1366'
feature_size = sp.load_npz(f"datasets/{dataset}/features.npz").shape[1]
hidden_size = 512
g_func = 'maxmin'
split = 'train'
batch_size = 64
epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize model
model = NHP(feature_size, hidden_size, g_func)

# load the dataset
train_hypergraph = Hypergraph(dataset, split, batch_size)
test_hypergraph = Hypergraph(dataset, "test", batch_size)
train_data = DataLoader(train_hypergraph, batch_size=1, shuffle=False)
test_data = DataLoader(test_hypergraph, batch_size=1, shuffle=False)

trained_model = train(model, train_data, epochs, device, learning_rate)
test_model(trained_model, test_data, device)
