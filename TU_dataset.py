import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from libs.utils import G2N2design
from libs.gnn import G2N2, get_n_params
import pandas as pd

from tqdm import tqdm

from torch_geometric.datasets import TUDataset
from torchmetrics.classification import BinaryAccuracy, Accuracy
from sklearn.model_selection import KFold


"""
hyperparamètres
"""


lr_init = 0.001
patience = 4
step = .90
epsi = 1e-6
ep = 150
batch_size=16


ntask = 0

operator = 'adj'

output_dim = 1
num_layer = 3
nodes_dim = [16,16,16] 
edges_dim = [16,16,16]
decision_depth = 3
final_neuron = [256,128]
readout_type  = "sum"
level = "graph"

"""
"""


transform = G2N2design( operator = operator)




root = "dataset/TUdataset"
name = "MUTAG"


dataset = TUDataset(root=root, name=name,pre_transform=transform,use_edge_attr=True)

dataset.data.y = dataset.data.y.type(torch.float32)


k_folds=10



        
def train():
    model.train()
    
    
    L=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)
        
        lss= loss(pre,data.y.reshape(pre.shape))
        
    
    
        lss.backward()
        
        optimizer.step()  
        L+=lss.item()

    return L/ntrid

def test():
    model.eval()
    yhat=[]
    ygrd=[]
    L=0
    for data in test_loader:
        data = data.to(device)

        pre=model(data)
        yhat.append(torch.sigmoid(pre).cpu().detach())
        ygrd.append(data.y.cpu().detach().reshape(pre.shape).type(torch.int32))
        lss= loss(pre,data.y.reshape(pre.shape))

        
        L+=lss.item()
    yhat=torch.cat(yhat,0)
    ygrd=torch.cat(ygrd,0)
    testmae=metric(yhat,ygrd)

     
    return L/ntsid,testmae 





kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print("training on fold number : ", fold)
    lr = lr_init

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_ids)



    node_input_dim = dataset.num_features
    edge_input_dim = dataset.num_edge_features


    ntrid = len(train_loader)
    
    ntsid = len(test_loader)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # select your model
    model = G2N2(node_input_dim, edge_input_dim, output_dim, device,
                 num_layer = num_layer, nodes_dim = nodes_dim, edges_dim = edges_dim,
                 decision_depth = decision_depth,final_neuron = final_neuron,
                 readout_type  = readout_type ,level = level).to(device)  
    param = get_n_params(model)
    print('number of parameters:', param)
       
                    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # For iMDB multi comment the following two lines
    
    loss = torch.nn.BCEWithLogitsLoss(reduction ='sum')
    metric = BinaryAccuracy()
    
    
    # For iMDB multi uncomment the following two lines
    
    # loss = torch.nn.CrossEntropyLoss(reduction ='sum')
    # metric = Accuracy(task ='multiclass',num_classes = 3)


    best_test=100000000000
    
    btest=0
    btestmae=0
    Train_loss = []
    #Val_loss = []
    Test_loss = []
    
    count = 0
    Test_mae = []
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
        test_loss,testmae = test()
        Train_loss.append(trloss)
        
        Test_loss.append(test_loss)
        Test_mae.append(testmae.numpy())
        if test_loss<best_test:
            torch.save(model.state_dict(), "save/"+name+str(fold)+"labels.dat")
            
            best_test=test_loss
            
            btestmae=testmae
            
            count = 0
        else:
            count +=1
        bTest_mae.append(btestmae.numpy())
        print('Epoch: {:02d}\nlr: {:.6f}, trloss: {:.6f},Testloss: {:.6f}, best test loss: {:.6f}, bestaccuracy:{:.6f}'.format(epoch,lr,trloss,test_loss,best_test,btestmae))
        
        
        
    
        results = { "testloss" : Test_loss,"testacc" : Test_mae,"btestacc" : bTest_mae
                   }
        
        
        results_df = pd.DataFrame(results)
        
        results_df.to_csv("data/"+name+str(fold)+"_predonly_GMN.dat", header = True, index = False)

