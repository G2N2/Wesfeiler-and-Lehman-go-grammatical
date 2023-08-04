
import torch
import torch.nn as nn

from torch_geometric.data import DataLoader


import numpy as np
from libs.utils import G2N2design
from libs.gnn import G2N2, get_n_params



import matplotlib.pyplot as plt
from libs.utils import TwoDGrid30
from sklearn.metrics import r2_score


affich = 50
"""
hyperparamètres
"""


lr = 0.005
patience = 100
step = .95
seuil = 1e-6
ep = 5000
batch_size=32


# ntask lowpass:0, highpass:1, bandpass:2 
ntask=2
task = ['lowpass','highpass','bandpass']

operator = 'adj'

output_dim = 1
num_layer = 5
nodes_dim = [2]*num_layer 
edges_dim = [4]*num_layer
decision_depth = 3
final_neuron = [128,64]
readout_type  = "sum"
level = "node"

"""
"""

transform = G2N2design(operator = operator)
dataset = TwoDGrid30(root="dataset/TwoDGrid30/",pre_transform=transform)


train_loader = DataLoader(dataset[0:1], batch_size=1, shuffle=False)
test_loader = DataLoader(dataset[1:2], batch_size=1, shuffle=False)
val_loader = DataLoader(dataset[2:3], batch_size=1, shuffle=False)

size = int(np.sqrt(dataset.data.x.shape[0]/3))


node_input_dim = dataset.num_features
edge_input_dim = dataset.num_edge_features



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# select your model
model = G2N2(node_input_dim, edge_input_dim, output_dim, device,
             num_layer = num_layer, nodes_dim = nodes_dim, edges_dim = edges_dim,
             decision_depth = decision_depth,final_neuron = final_neuron,
             readout_type  = readout_type ,level = level, relu = False).to(device)
param = get_n_params(model)




optimizer = torch.optim.Adam(model.parameters(), lr=lr)



def visualize(tensor,epoch,typ,n=30):
    y=tensor.detach().cpu().numpy()
    y=np.reshape(y,(n,n))
    plt.imshow(y[2:n-2,2:n-2].T);plt.colorbar()
    # plt.imshow(y.T);plt.colorbar()
    plt.title(f'Epoch: {epoch:02d}' + typ)
    plt.show()


def train(epoch):
    model.train()
    L=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data) 
        lss= torch.square((pre- data.y[:,ntask:ntask+1].reshape(pre.shape))).sum() 
        lss.backward()
        optimizer.step()

        a=pre[data.mask==1]
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1]
        
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())
        L+=lss.item()
        # if you want to see the image that GNN  produce
        if epoch%affich == 0:
            visualize(pre,epoch,'train',n=size)
    return L,r2

def test():
    model.eval()
    L=0;vL=0
    for data in test_loader:
        data = data.to(device)
        optimizer.zero_grad()        
        pre=model(data)
        lss= torch.square(data.mask*(pre- data.y[:,ntask:ntask+1].reshape(pre.shape))).sum() 
        L+=lss.item()
        a=pre[data.mask==1]
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1] 
        
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())
        if epoch%affich == 0:
            visualize(pre,epoch,'test',n=size)

    for data in val_loader:
        data = data.to(device)
        optimizer.zero_grad()        
        pre=model(data)        
        lss= torch.square(data.mask*(pre- data.y[:,ntask:ntask+1])).sum() 
        vL+=lss.item()
        a=pre[data.mask==1]   
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1] 
        vr2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())        

        # if you want to see the image that GNN  produce
        if epoch%affich == 0:
            visualize(pre,epoch,'val',n=size)
    return L,r2,vL,vr2

bval=-100
btest=0
count = 0

for epoch in range(1, ep):
    if count > patience:
        count = 0
        for g in optimizer.param_groups:
            lr = lr*step
            g['lr']= lr
    if lr < seuil:
        break
    trloss ,tr2   =train(epoch)
    test_loss,r2,vallos,vr2= test()
    count += 1 
    if bval<vr2:
        count = 0
        bval=vr2
        btest=r2
        torch.save(model.state_dict(), "save/"+task[ntask]+operator+"weight.dat")
   
    print('Epoch: {:02d}, lr : {:.4f}\n  trloss: {:.4f}, r2: {:.4f},valloss: {:.4f}, valr2: {:.4f},testloss: {:.4f}, bestestr2: {:.8f},{:.8f}'.format(epoch, lr,trloss,tr2,vallos,vr2,test_loss,r2,btest))
    