# test.py
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def test_model(model, test_loader):
    model.eval()
    pos_list = []
    neg_list = []

    for data in test_loader:
        with torch.no_grad():
            pos_score, neg_score = model(data)
            pos_list.append(pos_score.squeeze())
            neg_list.append(neg_score.squeeze())

    pos_scores = torch.cat(pos_list)
    neg_scores = torch.cat(neg_list)

    predictions = torch.cat([pos_scores, neg_scores]).numpy()
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).numpy()

    # calculate AUC
    auc = roc_auc_score(labels, predictions)
    # print(f"AUC: {auc: .2f}")

    # calculate Recall@k
    k = len(neg_scores) // 2
    top_k_indices = np.argsort(predictions)[::-1][:k]
    recall_at_k = np.sum(labels[top_k_indices]) / len(pos_scores)
    # print(f"Recall@k: {recall_at_k: .2f}")
    # print("Note: k is set to half of the number of positive scores")
    return auc, recall_at_k
