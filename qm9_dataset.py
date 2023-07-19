import torch
import torch.nn as nn
from prepareqm9dataset import QM9
from torch_geometric.loader import DataLoader
import numpy as np
from libs.utils import G2N2design
from libs.gnn import G2N2, get_n_params
import pandas as pd
import os

from tqdm import tqdm
from time import time

if not os.path.exists('save'):
   os.makedirs('save')
if not os.path.exists('dataset'):
   os.makedirs('dataset')
if not os.path.exists('data'):
   os.makedirs('data')

"""
hyperparamèters

lr : learning rate
patience : patience of the reduce on plateau
step : step of the reduce on plateau
epsi : threshold
ep : nb of epoch

"""

 


lr = 0.001
patience = 4
step = .90
epsi = 1e-6
ep = 500
batch_size=32

"""
ntask : 0 to 11, wich target to predict
"""
ntask = 0
"""
operator : chose wich matrix to feed G2N2
"""
operator = 'adj'

"""
Hyperparameters of G2N2 architechture
"""

output_dim = 1
num_layer = 3
nodes_dim = [32,32,32] 
edges_dim = [32,32,32]
decision_depth = 3
final_neuron = [512,256]
readout_type  = "sum"
level = "graph"

"""
"""


transform = G2N2design( operator = operator,QM9 = True)



dataset = QM9(root="dataset/QM9/",pre_transform=transform)


train_split = int(len(dataset)*.8)
valid_split = int(len(dataset)*.1)
test_split = len(dataset)-train_split-valid_split



train_dt, valid_dt, test_dt = torch.utils.data.random_split(dataset,[train_split,valid_split,test_split],
                                                            generator = torch.Generator().manual_seed(448))


ntrid = train_split
nvlid = valid_split
ntsid = test_split

train_loader = DataLoader(dataset[train_dt.indices], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset[valid_dt.indices], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[test_dt.indices], batch_size=batch_size, shuffle=False)

node_input_dim = dataset.num_features
edge_input_dim = dataset.num_edge_features



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = G2N2(node_input_dim, edge_input_dim, output_dim, device,
             num_layer = num_layer, nodes_dim = nodes_dim, edges_dim = edges_dim,
             decision_depth = decision_depth,final_neuron = final_neuron,
             readout_type  = readout_type ,level = level).to(device)




optimizer = torch.optim.Adam(model.parameters(), lr=lr)

param = get_n_params(model)
print('number of parameters:', param)


def train():
    model.train()
    
    L=0
    for data in (train_loader):

        data = data.to(device)
        
        
        pre=model(data)
        
        lss= torch.abs(pre- data.y[:,ntask:ntask+1]).sum() 
        
        for param in model.parameters():
            param.grad = None
        lss.backward()
        
        optimizer.step()
        # print(model.fc2.weight)
        L+=lss.item()
        
        
    
    return L/ntrid

def test():
    model.eval()
    yhat=[]
    ygrd=[]
    L=0
    for data in (test_loader):

        data = data.to(device)
        
        pre=model(data)
       
        yhat.append(pre.cpu().detach())
        ygrd.append(data.y[:,ntask:ntask+1].cpu().detach())
        lss= torch.abs(pre- data.y[:,ntask:ntask+1]).sum()
    
            
        L+=lss.item()
    yhat=torch.cat(yhat,0)
    ygrd=torch.cat(ygrd,0)
    testmae=np.abs(ygrd.numpy()-yhat.numpy()).mean()

    Lv=0
    for data in (val_loader):

        data = data.to(device)
        pre=model(data)
        lss= torch.abs(pre- data.y[:,ntask:ntask+1]).sum() 
        Lv+=lss.item()    
    return L/ntsid, Lv/nvlid,testmae


bval=100000000000

btest=0
btestmae=0
Train_loss = []
Val_loss = []
Test_loss = []

count = 0
bTest_mae = []
for epoch in tqdm(range(1, ep+1)):
    if count > patience:
        count = 0
        for g in optimizer.param_groups:
            lr = lr*step
            g['lr']= lr
    if lr < 1e-6:
        break
            
    trloss=train()
    test_loss,val_loss,testmae = test()
    Train_loss.append(trloss)
    Val_loss.append(val_loss)
    Test_loss.append(test_loss)
    if bval>val_loss:
        torch.save(model.state_dict(), "save/QM912labels.dat")
        
        bval=val_loss
        btest=test_loss
        btestmae=testmae
        # bTest_mae.append(btestmae*std[ntask].numpy())
        bTest_mae.append(btestmae)
        count = 0
    else:
        count +=1

    print('Epoch: {:02d}\nlr: {:.6f}, trloss: {:.6f},  Valloss: {:.6f},Testloss: {:.6f}, best test loss: {:.6f}, bestmae:{:.6f}'.format(epoch,lr,trloss,val_loss,test_loss,btest,btestmae))
    
    
    

    results = { "btestmae" : bTest_mae,
               }
    
    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv("data/QM9_"+str(ntask)+".dat", header = True, index = False)
