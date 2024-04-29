import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader import Hypergraph
from model_NHP import NHP


def train(model, dataloader, epochs, device, lr):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            pos_score = outputs['pos_score']
            neg_score = outputs['neg_score']

            loss = ranking_loss(pos_score, neg_score)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")
    return model


def ranking_loss(pos_score, neg_score):
    # encourage the model to assign higher scores to actual hyperlinks in E
    # compared to the average score of non-hyperlinks in F
    neg_mean = torch.mean(neg_score)
    loss = torch.mean(non_desc_func(neg_mean - pos_score))
    return loss


def non_desc_func(x):
    # non-decreasing function for loss function: log(1 + exp(x))
    return torch.log1p(torch.exp(x))


def test_train():
    # define parameters
    feature_size = 10
    hidden_size = 512
    g_func = 'maxmin'
    dataset = 'iAF1260b'
    split = 'train'
    batch_size = 5
    epochs = 5
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = NHP(feature_size, hidden_size, g_func)

    # load the dataset
    hypergraph_dataset = Hypergraph(dataset, split, feature_size, batch_size)
    dataloader = DataLoader(hypergraph_dataset, batch_size=1, shuffle=False)

    trained_model = train(model, dataloader, epochs, device, learning_rate)
    print(trained_model)
