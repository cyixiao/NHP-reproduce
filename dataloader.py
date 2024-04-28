import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


class Hypergraph(Dataset):
    def __init__(self, dataset, split, feature_size, batch_size):
        file_path = f"datasets/{dataset}/adjacency_matrix/indices.pkl"
        with open(file_path, 'rb') as file:
            indices_data = pickle.load(file)

        self.vertices_num = indices_data['vertices_num']
        self.hyperlinks_num = indices_data['hyperlinks_num']

        # initialize vertices features
        features = torch.zeros(self.vertices_num, feature_size)
        torch.nn.init.xavier_normal_(features)

        # get indices for positive and negative
        pos_indices = indices_data[split + '_hyperedges_i']
        neg_indices = indices_data[split + '_negative_i']

        # load features for vertices
        self.pos_features = load_features(features, pos_indices)
        self.neg_features = load_features(features, neg_indices)

        # load normalized tensor adj matrix
        pos_M = load_matrix(dataset, split, "hyperedges")
        neg_M = load_matrix(dataset, split, "negative")

        # feature aggregation, update features based on features from neighbors
        self.pos_matrix = torch.sparse.mm(pos_M, self.pos_features)
        self.neg_matrix = torch.sparse.mm(neg_M, self.neg_features)

        # set batch index
        self.batch = []
        last_index = -1
        distinct_count = 0

        for k in range(len(pos_indices)):
            hyperlink_index = pos_indices[k][0]
            if hyperlink_index != last_index:
                if distinct_count == batch_size:
                    self.batch.append(k)
                    distinct_count = 0
                last_index = hyperlink_index
                distinct_count += 1

        # ensure the last batch is in the array
        if self.batch[-1] != len(pos_indices):
            self.batch.append(len(pos_indices))

        self.pos_batch = get_batch_mask(pos_indices, batch_size)

    def __getitem__(self, index):
        start_index = self.batch[index]
        end_index = self.batch[index + 1]

        return {
            "batch_mask": self.pos_batch[start_index:end_index],
            "pos_features": self.pos_features[start_index:end_index],
            "neg_features": self.neg_features[start_index:end_index],
            "pos_matrix": self.pos_matrix[start_index:end_index],
            "neg_matrix": self.neg_matrix[start_index:end_index]
        }

    def __len__(self):
        return len(self.batch) - 1


def get_batch_mask(indices, batch_size):
    batch_mask = torch.zeros((len(indices), batch_size))
    for k in range(len(indices)):
        batch_index = indices[k][0] % batch_size
        batch_mask[k, batch_index] = 1
    return batch_mask


def load_matrix(dataset, split, link):
    matrix_path = f"datasets/{dataset}/adjacency_matrix/matrix_{split}_{link}.npz"
    matrix = sp.load_npz(matrix_path)
    return matrix_to_tensor(symmetric_normalization(matrix))


def load_features(features, indices):
    loaded_features = np.zeros((len(indices), features.shape[1]))
    for index, (_, vertex_index) in enumerate(indices):
        loaded_features[index] = features[vertex_index].numpy()
    return torch.from_numpy(loaded_features).float()


def symmetric_normalization(matrix):
    degrees = np.array(matrix.sum(axis=1)).flatten()
    # compute D^(-1/2)
    degrees_inv_sqrt = 1. / np.sqrt(degrees)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0.
    d_inv_sqrt = sp.diags(degrees_inv_sqrt)
    # return D^(-1/2) * M * D^(-1/2)
    return d_inv_sqrt @ matrix @ d_inv_sqrt


# https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
def matrix_to_tensor(matrix):
    matrix_coo = matrix.tocoo()
    values = matrix_coo.data
    indices = np.vstack((matrix_coo.row, matrix_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix_coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def test_dataloader():
    # dataloader test
    test_dataset = 'iAF1260b'
    test_split = 'train'
    test_feature_size = 10
    test_batch_size = 5

    hypergraph_dataset = Hypergraph(test_dataset, test_split, test_feature_size, test_batch_size)
    dataloader = DataLoader(hypergraph_dataset, batch_size=1, shuffle=False)

    for _i, data in enumerate(dataloader):
        print(f"Batch {_i + 1}:")
        print(data)
        if _i == 1:
            break
