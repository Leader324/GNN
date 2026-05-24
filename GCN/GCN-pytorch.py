import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
device=torch.device("cuda")

def load_cora(path_content="cora/cora.content",path_cites="cora/cora.cites"):
  
    '''Read nodes: feature+label'''
    content = pd.read_csv(path_content,sep="\t",header=None)        #Read the paper info: ID+feature+category
    ID=content.iloc[:,0].values     #ID
    X=torch.tensor(content.iloc[:,1:-1].values,dtype=torch.float32)     #features
    class_name=content.iloc[:,-1].values
    classes=sorted(set(class_name))
    class_to_index={name:i for i,name in enumerate(classes)}        #Turn categories into numbers in 0~6
    y=torch.tensor([class_to_index[name] for name in class_name],dtype=torch.long)      #labels
    n=X.shape[0]        #Total number of nodes
  
    '''Build the map from ID to corresponding nodes'''
    ID_to_index={pid:i for i,pid in enumerate(ID)}
  
    '''Build the adjacency matrix'''
    cites=pd.read_csv(path_cites,sep="\t",header=None)
    A=torch.zeros(n,n)
    for pid1,pid2 in cites.values:
        idx1=ID_to_index[pid1]
        idx2=ID_to_index[pid2]
        A[idx1,idx2]=1
        A[idx2,idx1]=1
      
    '''Devide the dataset'''
    train_mask=torch.zeros(n,dtype=torch.bool).to(device)
    val_mask=torch.zeros(n,dtype=torch.bool).to(device)
    test_mask=torch.zeros(n,dtype=torch.bool).to(device)
    train_mask[:140]=True
    val_mask[140:640]=True
    test_mask[640:1460]=True
    X,A,y=X.to(device),A.to(device),y.to(device)
    return X,A,y,train_mask,val_mask,test_mask
  
def load_citeseer(path_content="citeseer/citeseer.content",path_cites="citeseer/citeseer.cites"):
  
    '''Read nodes: feature+label'''
    content = pd.read_csv(path_content,sep="\t",header=None,dtype={0:str})        #Read the paper info: ID+feature+category
    ID=content.iloc[:,0].values     #ID
    X=torch.tensor(content.iloc[:,1:-1].values,dtype=torch.float32)     #features
    class_name=content.iloc[:,-1].values
    classes=sorted(set(class_name))
    class_to_index={name:i for i,name in enumerate(classes)}        #Turn categories into numbers in 0~6
    y=torch.tensor([class_to_index[name] for name in class_name],dtype=torch.long)      #labels
    n=X.shape[0]        #Total number of nodes
  
    '''Build the map from ID to corresponding nodes'''
    ID_to_index={pid:i for i,pid in enumerate(ID)}
  
    '''Build the adjacency matrix'''
    cites=pd.read_csv(path_cites,sep="\t",header=None)
    A=torch.zeros(n,n)
    valid_pids=set(ID)
    for pid1,pid2 in cites.values:
        if pid1 in valid_pids and pid2 in valid_pids:
            idx1=ID_to_index[pid1]
            idx2=ID_to_index[pid2]
            A[idx1,idx2]=1
            A[idx2,idx1]=1
          
    '''Devide the dataset'''
    train_mask=torch.zeros(n,dtype=torch.bool).to(device)
    val_mask=torch.zeros(n,dtype=torch.bool).to(device)
    test_mask=torch.zeros(n,dtype=torch.bool).to(device)
    train_mask[:120]=True
    val_mask[120:620]=True
    test_mask[620:1620]=True
    X,A,y=X.to(device),A.to(device),y.to(device)
    return X,A,y,train_mask,val_mask,test_mask
  
class GCN_Layer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(GCN_Layer, self).__init__()
        self.W = nn.Linear(in_dim,out_dim)      #Initialise W
      
    def normalise(self,A):      #Calculate DAD
        N=A.shape[0]
        I=torch.eye(N,device=A.device)      #Identity matrix
        A=A+I       #Reflexiv adjacency matrix
        D=A.sum(dim=1)
        D=torch.diag(1.0/torch.sqrt(D+1e-8))
        A_hat=D@A@D     #DAD, which is classic
        return A_hat
      
    def forward(self,X,A):      #Single layer GCN
        A_hat=self.normalise(A)
        return A_hat@self.W(X)
      
class GCN(nn.Module):    #Put the layers together
    def __init__(self,in_dim,hidden_dim,out_dim):
        super(GCN, self).__init__()
        self.gc1=GCN_Layer(in_dim,hidden_dim)
        self.gc2=GCN_Layer(hidden_dim,out_dim)
        self.Dropout=nn.Dropout(0.5)
      
    def forward(self,X,A):
        x=self.gc1(X,A)
        x=self.Dropout(x)
        x=F.relu(x)
        x=self.gc2(x,A)
        return x
      
    def model_train(self,X,A,y,train_mask,val_mask):
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(self.parameters(),lr=0.01,weight_decay=5e-4)
        train_loss=[]
      
        val_acc_history=[]
        #Early stop
        best_val_acc=0.0
        patience=10
        counter=0
        best_model_state=self.state_dict()
      
        for epoch in range(200):
            optimizer.zero_grad()
            y_pred=self.forward(X,A)
            loss=criterion(y_pred[train_mask],y[train_mask])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
          
            #Validate
            self.eval()
            with torch.no_grad():
                val_pred=self.forward(X,A)
                val_acc=self.accuracy(X,A,y,val_mask)
                val_acc_history.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc=val_acc
                counter=0
                best_model_state=self.state_dict()
            else:
                counter+=1
                if counter>=patience:
                    break
            if epoch % 10 == 0:
                print(f"epoch:{epoch + 1},loss:{loss.item()},val_acc:{val_acc}")
            self.train()
        return train_loss,val_acc_history
      
    def accuracy(self,X,A,y,mask):
        self.eval()
        with torch.no_grad():
            logits=self.forward(X,A)
        pred=logits.argmax(1)
        correct=torch.sum(pred[mask==1].eq(y[mask==1])).item()
        return correct/len(y[mask==1])
      
def main():
    choose=input("1 to run Cora, 2 to run citeseer\n")
    if choose=="1":
        X,A,y,train_mask,val_mask,test_mask=load_cora()
        model=GCN(1433,16,7).to(device)
    if choose=="2":
        X,A,y,train_mask,val_mask,test_mask=load_citeseer()
        model=GCN(3703,16,6).to(device)
      
    train_loss_history,val_acc_history=model.model_train(X,A,y,train_mask,val_mask)
    model.eval()
    with torch.no_grad():
        train_acc=model.accuracy(X,A,y,train_mask)
        val_acc=model.accuracy(X,A,y,val_mask)
        test_acc=model.accuracy(X,A,y,test_mask)
    print(f"train_acc:{train_acc},val_acc:{val_acc},test_acc:{test_acc}")
  
if __name__=='__main__':
    main()
