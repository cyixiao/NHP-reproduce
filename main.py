import torch
from torch.utils.data import DataLoader

from dataloader import Hypergraph
from model_NHP import NHP
from test import test_model
from train import train

feature_size = 32
hidden_size = 512
g_func = 'maxmin'
dataset = 'iAF1260b'
split = 'train'
batch_size = 64
epochs = 50
learning_rate = 0.001
k = 700
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize model
model = NHP(feature_size, hidden_size, g_func)

# load the dataset
train_hypergraph = Hypergraph(dataset, split, feature_size, batch_size)
test_hypergraph = Hypergraph(dataset, "test", feature_size, batch_size)
train_data = DataLoader(train_hypergraph, batch_size=1, shuffle=False)
test_data = DataLoader(test_hypergraph, batch_size=1, shuffle=False)

trained_model = train(model, train_data, epochs, device, learning_rate)
test_model(trained_model, test_data, device)
