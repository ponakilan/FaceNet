from . import accuracy_score


def train_one_epoch(model, optimizer, loss_fn, dataloader, device):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataloader):
        anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()

        outputs = model(anchor, positive, negative)

        loss = loss_fn(outputs[0], outputs[1], outputs[2])
        loss.backward()

        pos_dist, neg_dist, accuracy = accuracy_score(outputs[0], outputs[1], outputs[2])

        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print('  batch {} loss: {} accuracy: {}'.format(i + 1, last_loss, accuracy))
            running_loss = 0.

        return model, pos_dist, neg_dist, last_loss, accuracy
