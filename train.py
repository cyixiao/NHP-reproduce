import torch
from torch.optim import Adam


def train(model, dataloader, epochs, device, lr):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data in dataloader:
            optimizer.zero_grad()
            pos_score, neg_score = model(data)
            loss = ranking_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")
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
