import torch
import numpy as np


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def get_distance(anchor, positive, negative):
    anchor, positive, negative = anchor.detach().cpu(), positive.detach().cpu(), negative.detach().cpu()
    pos_dist = np.linalg.norm(anchor[0] - positive[0])
    neg_dist = np.linalg.norm(anchor[0] - negative[0])
    return pos_dist, neg_dist


def train_one_epoch(model, optimizer, lr, momentum, margin, train_loader, epoch, log_interval, device, run=None):
    model.train(True)

    # Instantiate the optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    # Instantiate the loss function
    loss_fn = TripletLoss(margin=margin)

    running_loss = 0.
    last_loss = 0.
    for i in range(len(train_loader)):
        data = next(iter(train_loader))
        anchor, positive, negative = data
        anchor = model(anchor.to(device))
        positive = model(positive.to(device))
        negative = model(negative.to(device))

        optimizer.zero_grad()

        loss = loss_fn(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_interval == 0:
            last_loss = running_loss / log_interval
            pos_dist, neg_dist = get_distance(anchor, positive, negative)
            if run is not None:
                run.log({"Training_loss": last_loss, "Positive_dist": pos_dist, "Negative_dist": neg_dist})
            print(f'\tBatch: {i}, Loss: {last_loss:.4f}')
            torch.save(model.state_dict(), f"model_{epoch}_{i}_{last_loss}_{pos_dist}_{neg_dist}.pt")
            running_loss = 0.

    return model, last_loss
