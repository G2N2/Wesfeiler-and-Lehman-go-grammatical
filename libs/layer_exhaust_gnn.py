from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from time import time




def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)





class Conv_agg(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(Conv_agg, self).__init__()

        assert K > 0       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapetensor = torch.zeros((K,1)).to(device)
          

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
       
        if bias:
           self.bias = Parameter(torch.Tensor(out_channels))
        else:
           self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, h, X,edge_index,batch_node):
        """"""
        
        zer = torch.unsqueeze(batch_node*0.,0)

        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
        resx[:,edge_index[0],edge_index[1]] = X.T
        res = torch.matmul(resx,h)
        res = torch.matmul(res,self.weight).sum(0)           

        if self.bias is not None:
            res += self.bias

        return res
    
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))
    
class Conv_agg_raw(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(Conv_agg_raw, self).__init__()

        assert K > 0       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapetensor = torch.zeros((K,1)).to(device)
          

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
       
        if bias:
           self.bias = Parameter(torch.Tensor(out_channels))
        else:
           self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, h, X,edge_index,batch_node):
        """"""
        
        zer = torch.unsqueeze(batch_node*0.,0)

        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
        resx[:,edge_index[0],edge_index[1]] = X.T
        res = torch.matmul(h,resx)
        res = torch.matmul(self.weight,res).sum(0)           

        if self.bias is not None:
            res = res.T +self.bias

        return res.T
    
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))


class G2N2Layer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,nnodeinput,nnodeoutput,device):
        super(G2N2Layer, self).__init__()

        self.nedgeinput = nedgeinput
        self.nnodeinput = nnodeinput
        self.nnodeoutput = nnodeoutput
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)
        self.shapetensor2 = torch.zeros(max(nnodeinput,nedgeinput),1).to(device)
         
        
        

        self.L1 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L2 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L3 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L4 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L5 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L6 = torch.nn.Linear(nnodeinput,max(nnodeinput,nedgeinput),bias = False)
        self.L7 = torch.nn.Linear(nnodeinput,max(nnodeinput,nedgeinput),bias = False)
        self.L8 = torch.nn.Linear(nnodeinput,max(nnodeinput,nedgeinput),bias = False)
        self.L9 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        self.L10 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        self.L11 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        self.L12 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        self.L13 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        self.L14 = torch.nn.Linear(nnodeinput,nnodeinput,bias = False)
        
        
        
        self.mlp1 = torch.nn.Linear(4*nedgeinput + 2*max(nnodeinput,nedgeinput) ,8*nedgeinput,bias=False)
        self.mlp2 = torch.nn.Linear(8*nedgeinput ,nedgeoutput,bias=False)
        
        self.mlp3 = torch.nn.Linear(3*nnodeinput ,8*nnodeinput,bias=False)
        self.mlp4 = torch.nn.Linear(8*nnodeinput ,nnodeoutput,bias=False)
        
        self.mlp5 = torch.nn.Linear(3*nnodeinput ,8*nnodeinput,bias=False)
        self.mlp6 = torch.nn.Linear(8*nnodeinput ,nnodeoutput,bias=False)
            
        
        self.agg = Conv_agg(nnodeinput,nnodeinput, device, K=nedgeoutput,bias = True)
        
        self.agg_raw = Conv_agg_raw(nnodeinput,nnodeinput, device, K=nedgeoutput,bias = True)

    
     
    
    def matmul(self,X,Y,batch_node,edge_index):
        
        zer = torch.unsqueeze(batch_node*0.,0).detach()

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()

        resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()
       
       
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        resy[:,edge_index[0],edge_index[1]] = Y.T
        
        res = torch.matmul(resx,resy)
        
        return res[:,edge_index[0],edge_index[1]].T
    
       
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[0],edge_index[1]].T
    
    def transpo(self,X,batch_node,edge_index):
        zer = torch.unsqueeze(batch_node*0.,0).detach()

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer).detach()
        resx[:,edge_index[1],edge_index[0]] = X.T
        
        return resx[:,edge_index[0],edge_index[1]].T
    
    def VcVr(self,x,v,batch_node,edge_index):
        zer = torch.unsqueeze(batch_node*0.,0).detach()

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor2,zer),2),zer).detach()
        
        
        resx[:,edge_index[0],edge_index[1]] = torch.matmul(x.T.unsqueeze(2),v.unsqueeze(1))[:,edge_index[0],edge_index[1]]
        
        return resx[:,edge_index[0],edge_index[1]].T
        
    

    def forward(self, x,v,edge_index,C,batch_node):
        
        
        
        tmp_diag = self.diag( (self.L6(x)/self.nnodeinput),edge_index)
        tmp_transpo = self.transpo(self.L5(C)/self.nedgeinput,batch_node,edge_index)
        tmp_vcvr = self.VcVr((self.L7(x)/self.nnodeinput),self.L8(v.T).T/self.nnodeinput,batch_node,edge_index)
        tmp_matmul = self.matmul(  (self.L3(C)/self.nedgeinput),  (self.L4(C)/self.nedgeinput),batch_node, edge_index)
        tmp=torch.cat([  (C),(self.L1(C)/self.nedgeinput)*  (self.L2(C)/self.nedgeinput),tmp_diag,tmp_matmul,tmp_transpo,tmp_vcvr],1)
        Cout = self.mlp2(torch.relu((self.mlp1(tmp))))
        
        xout=torch.cat([self.agg(x, Cout, edge_index, batch_node)/self.nnodeinput,self.L12(v.T)/self.nnodeinput,(self.L11(x)/self.nnodeinput)*(self.L12(x)/self.nnodeinput)],1)
        xout = self.mlp4(torch.relu((self.mlp3(xout))))
        
        vout=torch.cat([self.agg_raw(v, Cout, edge_index, batch_node)/self.nnodeinput,(self.L10(x)/self.nnodeinput).T,(self.L13(v.T).T/self.nnodeinput)*(self.L14(v.T).T/self.nnodeinput)],0)
        vout = self.mlp6(torch.relu((self.mlp5(vout.T)))).T
        
        return xout , vout,  Cout
    
