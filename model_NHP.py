import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import Hypergraph


class NHP(nn.Module):
    def __init__(self, feature_size, hidden_size, g_func):
        super(NHP, self).__init__()
        self.g_func = g_func
        # Self loop layer, retaining original information:
        self.self_loop = nn.Linear(feature_size, hidden_size)
        # Hyperlink-aware GCN layer:
        self.hyperlink_aware = nn.Linear(feature_size, hidden_size)
        self.activate = nn.ReLU()
        # Hyperlink scoring layer:
        self.hyperlink_score = nn.Linear(hidden_size, 1)

    def forward(self, data):
        # Hyperlink-aware GCN layer
        pos_loop = self.self_loop(data['pos_features'][0])
        pos_gcn = self.hyperlink_aware(data['pos_matrix'][0])
        pos_features = self.activate(pos_loop + pos_gcn)

        neg_loop = self.self_loop(data['neg_features'][0])
        neg_gcn = self.hyperlink_aware(data['neg_matrix'][0])
        neg_features = self.activate(neg_loop + neg_gcn)

        # apply scoring methods
        if self.g_func == 'mean':
            pos_score = torch.sigmoid(self.hyperlink_score(self._mean(data['batch_mask'][0], pos_features)))
            neg_score = torch.sigmoid(self.hyperlink_score(self._mean(data['batch_mask'][0], neg_features)))
        elif self.g_func == 'maxmin':
            pos_score = torch.sigmoid(self.hyperlink_score(self._maxmin(data['batch_mask'][0], pos_features)))
            neg_score = torch.sigmoid(self.hyperlink_score(self._maxmin(data['batch_mask'][0], neg_features)))
        else:
            raise ValueError("invalid scoring function")

        return {"pos_score": pos_score, "neg_score": neg_score}

    @staticmethod
    def _mean(batch, features):
        column_sums = torch.sum(batch, dim=0)
        norm = batch / column_sums
        return torch.mm(norm.t(), features)

    @staticmethod
    def _maxmin(batch, features):
        # max and min values
        num_hyperlinks = batch.size(1)
        num_features = features.size(1)

        max_features = torch.full((num_hyperlinks, num_features), float('-inf'))
        min_features = torch.full((num_hyperlinks, num_features), float('inf'))

        for i in range(num_hyperlinks):
            # get indices for this hyperlink
            indices = (batch[:, i] == 1).nonzero(as_tuple=True)[0]

            if indices.size(0) > 0:
                # get elementwise max and min
                hyperlink_features = features[indices, :]
                max_features[i, :] = torch.max(hyperlink_features, dim=0).values
                min_features[i, :] = torch.min(hyperlink_features, dim=0).values

        return max_features - min_features


def test_nhp_model():
    # define parameters
    feature_size = 10
    hidden_size = 512
    g_func = 'maxmin'
    dataset = 'iAF1260b'
    split = 'train'
    batch_size = 5

    # initialize model
    model = NHP(feature_size, hidden_size, g_func)
    model.eval()

    # load the dataset
    hypergraph_dataset = Hypergraph(dataset, split, batch_size)
    dataloader = DataLoader(hypergraph_dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(dataloader):
        if i == 1:
            break
        output = model(data)
        print(f"Batch {i + 1} output:", output)
