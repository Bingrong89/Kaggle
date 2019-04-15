#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Spyder Editor
"""

"""


"""


import os
import sys
import re
import operator 
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import support_functions as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()

class TrainDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe


    def __getitem__(self,idx):

        data = torch.tensor(self.data.iloc[idx][2:].tolist())              
        label = torch.tensor(self.data.iloc[idx][1],dtype=torch.long)
        return [data,label]
    
    def __len__(self):
        return len(self.data)

class ValDataset(Dataset):
    def __init__(self,dataframe):
        self.data = dataframe
    
    def __getitem__(self,idx):
        data = torch.tensor(self.data.iloc[idx][1:].tolist())
        iloc = idx
        
        return [data,iloc]
    
    def __len__(self):
        return len(self.data)


class Model1(nn.Module):
    def __init__(self):
        super(Model1,self).__init__()
        
        
        self.linear0 = nn.Linear(1,16)
        self.linear1 = nn.Linear(1600,1)
        
        self.bn1 = nn.BatchNorm1d(200,momentum=None,affine=False)
        self.avgpool1d = nn.AvgPool1d(3,2,1)


        
    def forward(self,x):
        
        x = self.bn1(x)  
 
        bs = x.shape[0]
        x = x.view(bs,200,1)
        output = F.relu(self.linear0(x))
  
        output=self.avgpool1d(output)
      
        output = output.view(bs,1600)
        
       # output = self.bn2(output)
        
       
        output = self.linear1(output)
    
        
        return output

def range_intervals(number,intv=0.2):
    if (1/intv).is_integer() == False:
        raise Exception("%.1d is an invalid interval division"% (1/intv))
    x = int(intv*number)
    listx=[]
    for z in range(1,int(1/intv)):
        listx.append(x*z)
    listx.append(number-1) #used in 0-index env...
    return listx

def train_model(model,batchtrain,batchtest,criterion,optimizer,num_epochs,logfile=None,alert=False):   
    
    if logfile != None:
        assert isinstance(logfile,str),'%s has to be a string'%logfile
        if os.path.isfile(logfile):
            raise Exception('%s Log File already exists'%logfile)        
        else:
            file_obj = open(logfile,'a')
    
    interval = 0.5
    save_intervals = range_intervals(len(batchtrain),interval)
    start_time = time.time()
    
    all_auc = {} #gather all the auc scores here
    save_name = ""
    for epoch in range(num_epochs):
        epoch_trainloss = 0
        epoch_itv_acc = 0
        epoch_trainstart = time.time()
        
        model.train()
        
        for batch_count, data in enumerate(tqdm(batchtrain,desc="Epoch %s Training Phase"%epoch)):
            traindata = data[0].to(device)
            trainlabel = data[1].to(device)
            optimizer.zero_grad()
            
            output = model(traindata)
            pred_scorez,pred_idx = output.max(1)
            output2 = output.squeeze().clone()
            trainloss = criterion(output2,trainlabel.float())
            trainloss.backward()
            
            nn.utils.clip_grad_value_(model.parameters(),0.8)
            optimizer.step()
            
            epoch_trainloss += trainloss.item()
            auc_score = roc_auc_score(trainlabel.cpu().numpy(),output2.detach().cpu().numpy())
            epoch_itv_acc += auc_score * 100
            
            if batch_count in save_intervals:
                percent_compl = (save_intervals.index(batch_count) +1) * (interval*100)
                
                train_statement = "(PHASE::TRAIN) Epoch_%i %i%% :: Avg Loss: %.3f | Acc: %.3f :: Time Taken: %.2fs"\
                %(epoch,percent_compl, epoch_trainloss/save_intervals[0], epoch_itv_acc/save_intervals[0],time.time()-epoch_trainstart)
                
                epoch_trainloss = 0
                epoch_itv_acc = 0
                epoch_trainstart = time.time()
                tqdm.write(train_statement +"\n")
                if logfile != None:
                    file_obj.write(train_statement +"\n")
                
                save_name = "Epoch%i_%i.pth"%(epoch,percent_compl)
                torch.save(model.state_dict(),save_name)
            
                test_acc = 0
                epoch_teststart = time.time()
                model.eval()
                
                with torch.no_grad():
                    for batch_count, data in enumerate(batchtest):
                        testdata = data[0].to(device)
                        testlabel = data[1].to(device)
                        
                        output = model(testdata)
                        pred_scorez,pred_idx = output.max(1)
                        #output2=output
                        output2 = output.squeeze().clone()                   
                        auc_score = roc_auc_score(testlabel.cpu().numpy(),(output2.detach().cpu()).numpy())                    
                        test_acc += auc_score* 100         
                        
                test_statement = "\n(PHASE::TEST) Epoch %i %i%% :::: Epoch Acc: %.4f%% ::Time Taken: %.2fs"\
                %(epoch,percent_compl,test_acc/len(batchtest),time.time()-epoch_teststart)
                tqdm.write(test_statement)
                all_auc[save_name] = test_acc/len(batchtest)
                if logfile!= None:
                    file_obj.write(test_statement +"\n")
        
    #get the best model based on allauc and do submission
    bestmodelname = list(all_auc.keys())[list(all_auc.values()).index(max(all_auc.values()))]
    modelpath = os.path.join(os.getcwd(),bestmodelname)
    model.load_state_dict(torch.load(modelpath))
    
    print("Using %s as the best model!"%bestmodelname)
    
    with torch.no_grad():
        valdata = pd.read_csv('test.csv')
        scaler = StandardScaler()
        valdata[valdata.columns[1:]] = scaler.fit_transform(valdata[valdata.columns[1:]])

        submitsheet = pd.DataFrame(valdata['ID_code'])
        submitsheet['target'] = 'None'
        
        valdataset = ValDataset(valdata)
        valbatch = DataLoader(valdataset,batch_size=32,drop_last=False)
                
        for data in tqdm(valbatch,desc="Creating Submission Results"):
            valdata = data[0].to(device)
            ilocs = data[1].to(device).squeeze()
           
            output = model(valdata)
            output = output.squeeze()

            for idx,index in enumerate(ilocs):
                submitsheet['target'].iloc[index.item()] = output[idx].cpu().numpy() 

        submitsheet.to_csv('submission.csv',index=False)   

    total_time = time.time() - start_time
    final_statement = "\nTotal Elapsed Time == % .2fs"%(total_time)
    tqdm.write(final_statement)
    if logfile!= None:
        file_obj.write(final_statement)
    if alert == True:
        sf.send_alert("Santander Model1 training has completed")

df = pd.read_csv('train.csv')
scaler = StandardScaler()
df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])

df2 = df[df['target'] == 1]
df3 = pd.concat([df2]*8)
df = df.append(df3)
df=shuffle(df)
df.reset_index(drop=True,inplace=True)

#df=shuffle(df)
splitnumber = int(0.8* len(df))
traindf = df[0:splitnumber]
testdf = df[splitnumber:]
"""
one_ratio = len(traindf[traindf['target'] == 1])/len(traindf[traindf['target'] == 0])
print("RATIO OF ONES TO ZEROS IS ::: %.3f"%one_ratio)
zero_ratio = 1 - one_ratio

sampler_weights = torch.zeros(len(traindf))
sampler2_weights = torch.zeros(len(testdf))

for x in range(len(traindf)):
    if traindf.iloc[x].target ==0:
        sampler_weights[x] = one_ratio
    elif traindf.iloc[x].target ==1:
        sampler_weights[x] = zero_ratio

for x in range(len(testdf)):
    if testdf.iloc[x].target ==0:
        sampler2_weights[x] = one_ratio
    elif testdf.iloc[x].target ==1:
        sampler2_weights[x] = zero_ratio      
        
#sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
#sampler2 = WeightedRandomSampler(sampler2_weights, len(sampler2_weights))
"""
traindataset= TrainDataset(traindf)
testdataset = TrainDataset(testdf)

trainbatch = DataLoader(traindataset,batch_size=32,drop_last=True)
testbatch = DataLoader(testdataset,batch_size=32,drop_last=True)

model = Model1().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adadelta(model.parameters())#,lr=0.001)
#optimizer = optim.SGD(model.parameters(),lr=0.01)

train_model(model,trainbatch,testbatch,criterion,optimizer,8)

    




