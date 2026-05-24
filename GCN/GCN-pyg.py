import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

device=torch.device("cuda")
dataset=Planetoid(root="./Planetoid/Cora",name="Cora")
data=dataset[0].to(device)

class GCN(nn.Module):
  
    def __init__(self,in_channel,hidden_channel,out_channel):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channel,hidden_channel)
        self.conv2 = GCNConv(hidden_channel,out_channel)
      
    def forward(self,x,edge_index):
        x=F.dropout(x,0.5)
        x=self.conv1(x,edge_index)
        x=F.dropout(x,0.5)
        x=self.conv2(x,edge_index)
        return x
      
    def model_train(self,data):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=0.01,weight_decay=5e-4)
        for epoch in range(200):
            optimizer.zero_grad()
            pred=self(data.x,data.edge_index)
            loss=criterion(pred[data.train_mask],data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"epoch:{epoch},loss:{loss}")
              
    def model_test(self,data):
        self.eval()
        with torch.no_grad():
            pred=self(data.x,data.edge_index).argmax(dim=1)
            acc=[]
            for mask in [data.train_mask,data.val_mask,data.test_mask]:
                acc.append(pred[mask].eq(data.y[mask]).float().mean())
        return acc

def main():
    model=GCN(data.num_features,16,7).to(device)
    model.model_train(data)
    acc=model.model_test(data)
    print(acc)
  
if __name__=='__main__':
    main()
