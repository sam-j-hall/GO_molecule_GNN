import torch
import torch.nn as nn

def test_model(loader, model, device):

    loss_all = 0
    model.eval()

    for batch in loader:

        batch = batch.to(device)

        batch_size = batch.y.shape[0] // 200
        batch.y = batch.y.view(batch_size, 200)

        with torch.no_grad():
            pred = model(batch)

        loss = nn.MSELoss()(pred.double(), batch.y.double())

        loss_all += loss.item() * batch.num_graphs

    return loss_all / len(loader.dataset) 