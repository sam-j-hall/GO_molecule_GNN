import torch.nn as nn

def train_model(epoch, loader, model, device, optimizer):

    loss_all = 0

    model.train()

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)

        batch_size = batch.y.shape[0] // 200
        batch.y = batch.y.view(batch_size, 200)

        loss = nn.MSELoss()(pred.double(), batch.y.double())

        loss.backward()

        loss_all += loss.item() * batch.num_graphs

        optimizer.step()

    return loss_all / len(loader.dataset)