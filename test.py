import torch
import torch.nn as nn

def test_model(loader, model, device):

    loss_all = 0
    model.eval()

    for batch in loader:

        batch = batch.to(device)

        batch_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(batch_size, 200)

        with torch.no_grad():
            pred = model(batch)

        loss = nn.MSELoss()(pred.double(), batch.spectrum.double())

        loss_all += loss.item() * batch.num_graphs

    return loss_all / len(loader.dataset)

def test_atom(loader, model, device):
    model.eval()
    loss_all = 0

    for batch in loader:
        batch = batch.to(device)
        #x, edge_index, index = batch.x, batch.edge_index, batch.index
        
        # Add batch dimension to index
        #batch_index = index.unsqueeze(1)
        batch_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(batch_size, 200)

        with torch.no_grad():
            #pred = model(batch)
            pred, node, select, tot = model(batch)
        
        #print(pred)
   #     alpha=10
       # loss = nn.MSELoss()(pred.view(-1, 1).double(),
        #                  batch.y.view(-1, 1).double()) 
        
        loss = nn.MSELoss()(pred.double(), batch.spectrum.double()) 
    #    +alpha*F.mse_loss(torch.log(pred.view(-1, 1).double()+0.001), torch.log(batch.y.view(-1, 1).double()+0.001))
       # loss=nn.SmoothL1Loss()(pred.view(-1, 1).double(), 
       #                 batch.y.view(-1, 1).double())
        loss_all += loss.item() * batch.num_graphs

    return loss_all / len(loader.dataset)