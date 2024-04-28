import torch
import torch.nn as nn

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
        max_features = torch.full_like(features, float('-inf'))
        min_features = torch.full_like(features, float('inf'))

        # for each column of features
        for i in range(features.size(1)):
            # mask select to get relevant features
            mask = torch.masked_select(features[:, i], batch.bool())
            # count num of features in each hyperlink
            count = batch.sum(1)
            # get max min value for each hyperlink
            for j in range(batch.size(0)):
                start = int(torch.sum(count[:j]))
                end = int(start + count[j])
                if end > start:
                    hyperlink_features = mask[start:end]
                    max_features[j, i] = torch.max(hyperlink_features)
                    min_features[j, i] = torch.min(hyperlink_features)

        return max_features - min_features
