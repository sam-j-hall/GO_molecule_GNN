import torch.nn as nn
import torch
import numpy as np

def train_model(epoch, loader, model, device, optimizer):

    loss_all = 0

    model.train()

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)

        batch_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(batch_size, 200)

        loss = nn.MSELoss()(pred.double(), batch.spectrum.double())
        #loss = sid(pred.double(), batch.spectrum.double())

        loss.backward()

        loss_all += loss.item() * batch.num_graphs

        optimizer.step()

    return loss_all / len(loader.dataset)

def sid(prediction, true):    

    sid = []

    for i in range(len(prediction)):
        temp = 0
        for x in range(200):
            pred = prediction[i][x].detach().numpy()
            tgt = true[i][x].detach().numpy()
            if pred < 0:
                pred = 0.0000000001

            temp += pred * np.log(pred/tgt) + tgt * np.log(tgt/pred)
            sid.append(temp)
    value = np.array(sum(sid) / len(sid))
    value = torch.autograd.Variable(torch.from_numpy(value), requires_grad=True)

    return value

def train_atom(epoch, loader, model, device, optimizer):
    model.train()
    loss_all = 0
    node_list = []
    select_list = []
    tot_list = []
    smiles_list = []
   # batch_size = 128

    for batch in loader:
        
        #print(batch.idx,batch.smiles)
        batch = batch.to(device)
        #x, edge_index,index = batch.x,batch.edge_index,batch.index
       
        #print(batch.idx,batch.atom_index)
        # Add batch dimension to index
        #batch_index = index.unsqueeze(1)

        optimizer.zero_grad()
        
        pred, select = model(batch)
        #pred = model(batch)
        batch_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(batch_size, 200)
        #pred = torch.clamp(pred, min=0.0)
        #print(batch.y.shape)
        #new_pred=pred.view(batch.y[:,0:99].shape)

        #pred=ss[0]
        #emb=ss[1]
        alpha = 10

        #loss = nn.MSELoss()(pred.view(-1, 1).double(), 
        #                batch.y[:,:100].view(-1, 1).double()) 
        
        loss = nn.MSELoss()(pred.double(), batch.spectrum.double()) 
       # + alpha*F.mse_loss(torch.log(pred.view(-1, 1).double()+0.001), torch.log(batch.y.view(-1, 1).double()+0.001))
        
        #p_loss=nn.SmoothL1Loss()(new_pred.double(), 
        #                batch.y.double())
        
        #loss=nn.SmoothL1Loss()(pred.view(-1, 1).double(), 
        #                batch.y.view(-1, 1).double())
        #print(loss)
        #print(pred.view(-1, 1).double(), batch.y.view(-1, 1).double())
       # pred_list.append(pred.view(-1, 1).double())
        #y_list.append(batch.y.view(-1, 1).double())
        
        loss.backward()
        #print(batch.num_graphs)
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
        
    #     if epoch == 999 :
    #         node_list.append(node)
    #         select_list.append(select)
    #         smiles_list.append(batch.smiles)
    #         tot_list.append(tot)

        out = pred[0].cpu().detach().numpy()
        true = batch.spectrum[0].cpu().detach().numpy()

    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # if epoch == 999:
    #     a = node_list
    #     b = select_list
    #     c = smiles_list
    #     d = tot_list
    
    #print(len(train_loader.dataset))
        #emb_list.append(emb)
    return loss_all / len(loader.dataset), out, true, select#, a ,b ,c ,d, out, true

def new_train(model, loader, optimizer, device):

    model.train()

    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        pred = model(data)
        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)

        loss = nn.MSELoss()(pred, data.spectrum)

        total_loss += loss.item() * data.num_graphs

        loss.backward()

        optimizer.step()

    return total_loss / len(loader.dataset)