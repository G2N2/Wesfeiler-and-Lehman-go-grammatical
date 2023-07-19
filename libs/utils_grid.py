import numpy as np
import torch
from torch_geometric.loader import DataLoader



def grid(n,start,end):
    l = []
    k = end-start +1
    for i in range(0,k**n):
        m = np.zeros(n, dtype = int)
        for j in range(1,n+1):
            tmp = k**(n-j)
            m[j-1] = int(2**(start + (i//tmp)%k))
        l.append(m)
    return l



def train(model, loader, optimizer, loss, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pre=model(batch)
        
        lss= loss(pre, batch.y) 
        
        lss.backward()
        optimizer.step()
        
def evaluate(model, loader, evaluator, device ):
    model.eval()
    yhat=[]
    ygrd=[]
    for data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            pre=model(data)
        yhat.append(pre.cpu().detach())
        ygrd.append(data.y.cpu().detach())
                 
    yhat=torch.cat(yhat).numpy()
    ygrd=torch.cat(ygrd).numpy()
    evaluate = evaluator(ygrd,yhat)

    return evaluate
    