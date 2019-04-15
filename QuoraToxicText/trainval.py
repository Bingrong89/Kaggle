import os
import sys
import re
import operator
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import support_functions as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab): #if word from vocab in embedindex, assign to a and add to k
        try:
            a[word] = embeddings_index[word]
            k += vocab[word] #vocab is a dictonary of word counts
        except:# if word not in embed index, execute this loop

            oov[word] = vocab[word]
            i += vocab[word]
            pass

def split_qmark(x): #need to insert space before question marks, else no embedding for word, took too long to this -_-
    #the tqmd progress bar turns ugly.. prob cause of recursion
    """Will fail if '?' is placed infront of word.. e.g:  '?hello'   , but im not expecting that for this task"""
    x = str(x)
    if len(x.strip(' ')) ==0:
        return x
    pos = x[1:].find('?') +1 #skip the first char, incase first char is '?' to not insert space before it
    if pos == 0:
        pass
    else:
        x = x[:pos] + " " + x[pos:]
    if pos+1!= len(x)-1:
        back = split_qmark(x[pos+1:])
        x = x[:pos+1] + back
    
    
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'What\'s' : 'Whats',
                'so,'     : 'so ,',
                'I’m'     : 'I am',
                'you\'ve' : 'you have',
                'isn\'t'  : 'is not',
                '"The'    : '" The',
                'don’t'   : 'don\'t',
                'aren\'t' : 'arent',
                'What’s' : 'Whats',
                'won\'t'  : 'wont',
                'me,'     : 'me ,',
                'they\'re': 'they are',
         #      '\('      : ' \(',
           #     '\(or'     : '\( or',
                'haven\'t': 'havent',
                'yes,'    : 'yes ,'
            #    '\( I'     : '\( I',
             #   '\( in'    : '\( in'    
                
                }
mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

class MakeDataset(Dataset):
    def __init__(self, data2make):
        self.data = data2make
      #  self.embedding = embedding
        
        
    def __getitem__(self,idx):
        whole_sent = self.data["question_text"][idx].split()
        label = self.data['target'][idx]
        return [whole_sent,label]
    
    def __len__(self):
        return len(self.data)


def sort_batch(batchdata):
    sorted_list = []
    for z in range(2):
        for sent in range(len(batchdata[0][0])):
            innerlist=[]
            if len(sorted_list) > len(batchdata[0][0]):
                break
            for batch in range(len(batchdata)):
                if z ==0:
                    innerlist.append(batchdata[batch][z][sent])
                else:
                    #print("triggered")
                    innerlist.append(batchdata[batch][z])
            
            sorted_list.append(tuple(innerlist))
        
    return(sorted_list)


def collater(input):
    listlength = len(input)
    maxlen = 0
    for x in input:
        if len(x[0])> maxlen:
            maxlen = len(x[0])

    for idx,x in enumerate(input):
        if len(x[0])<maxlen:
            diff = maxlen - len(x[0])
            addem = ['XNULLX'] * diff
            x[0].extend(addem)
    
    return sort_batch(input)

class Modelz(nn.Module):
    def __init__(self,hidden,embed):
        super(Modelz,self).__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(self.hidden,self.hidden)
        self.linear = nn.Linear(self.hidden,2)
        self.embed = embed
    
    def forward(self,x):
        
        sentencelength = len(x)-1 #exclude label tuple
        hc_state= None
        for numba in range(sentencelength):
            input = conv_inputtuple(x[numba],self.embed)
            output,hc_output = self.lstm(input,hc_state)
            hc_state = hc_output
    
        output = self.linear(output)    
        return output

def conv_inputtuple(inputtuple,embed):
    outputtensor = torch.empty(0,dtype=torch.float,device=device)
    for word in inputtuple:
        if word == 'XNULLX'or word not in embed:
            tensorz = torch.zeros(300,dtype=torch.float,requires_grad=False,device=device).view(1,1,-1) #actually req grad doesnt matter.. torch cat will push everything to True
        else:
            tensorz = torch.from_numpy(embed[word]).view(1,1,-1)
            tensorz.requires_grad = True
            tensorz = tensorz.to(device)
        outputtensor = torch.cat((outputtensor,tensorz),dim=1)
        outputtensor = outputtensor.to(device)
    return outputtensor    

def range_intervals(number,intv=0.2):
    if (1/intv).is_integer() == False:
        raise Exception("%.1d is an invalid interval division")
    x = int(intv*number)
    listx=[]
    for z in range(1,int(1/intv)):
        listx.append(x*z)
    listx.append(number-1) #used in 0-index env...
    return listx

def eval_batchacc(pred,gt):
    #gt is ndarray
    gtx = torch.tensor(gt,dtype=torch.long,device=device)
    pred = pred.squeeze(0)
    assert gtx.shape[0]==pred.shape[0],"Pred Size: %s .. GT Size %s . . .NOT SAME"%(pred.shape[0],gtx.shape[0])
    
    lengthz = pred.shape[0]
    accurate_preds = 0
    for x in range(lengthz):
        if pred[x] == gtx[x]:
            accurate_preds+=1
    
    accuracy = (accurate_preds/lengthz) * 100
    
    return accuracy

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
    
    for epoch in range(num_epochs):
        epoch_trainloss = 0
        epoch_itv_acc = 0
        epoch_trainstart = time.time()
        
        model.train()
        
        for batch_count, data in enumerate(tqdm(batchtrain,desc="Epoch %s Training Phase"%epoch)):
            traindata = data[:-1]#.to(device)
            trainlabel = data[-1]#.to(device)
            optimizer.zero_grad()
            
            output = model(traindata)
            pred_scorez,pred_idx = output.max(2)
            gt_idx = torch.tensor(trainlabel,dtype=torch.float,device=device)
            
            trainloss = criterion(pred_scorez.squeeze(0),gt_idx)
            trainloss.backward()
            optimizer.step()
            
            epoch_trainloss += trainloss.item()
            epoch_itv_acc += eval_batchacc(pred_idx,trainlabel)
            
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
            for batch_count, data in enumerate(tqdm(batchtest,desc="Epoch %s Testing Phase"%epoch)):
                testdata = data[:-1]#.to(device)
                testlabel = data[-1]#.to(device)
                
                output = model(testdata)
                pred_scorez,pred_idx = output.max(2)
                test_acc += eval_batchacc(pred_idx,testlabel)
                
            test_statement = "(PHASE::TEST) Epoch %i :::: Epoch Acc: %.2d%% ::Time Taken: %.2fs"\
            %(epoch,test_acc/len(batchtest),time.time()-epoch_teststart) 
            tqdm.write(test_statement)
            if logfile!= None:
                file_obj.write(test_statement +"\n")
                
    total_time = time.time() - start_time
    final_statement = "Total Elapsed Time == % .2fs"%(total_time)
    tqdm.write(final_statement)
    if logfile!= None:
        file_obj.write(final_statement)
    
    if alert == True:
        sf.send_alert("Toxic Network training has completed")




if __name__ == '__main__':
    
    torch.manual_seed(0)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        np.random.seed(0)
    except NameError:
        pass

    tqdm.pandas()
    
    base = os.getcwd()
    embedding = os.path.join(base,'Base','glove.840B.300d','glove.840B.300d.txt')
    trainfile = os.path.join(base,'Base','train.csv')
    alldata = pd.read_csv(trainfile)
    
    
    if 'glove.txt' not in os.listdir():
        glove2word2vec(embedding,"glove.txt")
    
    glove_embed = KeyedVectors.load_word2vec_format("glove.txt",binary=False)
    
    alldata["question_text"] = alldata["question_text"].progress_apply(lambda x: split_qmark(x))
    alldata["question_text"] = alldata["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    
    split_num = int(0.8*len(alldata))
    
    train = alldata[:split_num]
    test = alldata[split_num:]
    
    
    """
    sentences = train["question_text"].progress_apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab,glove_embed) #check coverage of embedding on training data
    """
    
    traindataset = MakeDataset(train)
    testdataset = MakeDataset(test)
    
    trainbatch = DataLoader(traindataset,batch_size=16,shuffle=True,collate_fn=collater)
    testbatch = DataLoader(testdataset,batch_size=16,shuffle=True,collate_fn=collater)
    
    model = Modelz(300,glove_embed).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adagrad(model.parameters(), lr = 0.01)
    
    try:
        train_model(model,trainbatch,testbatch,criterion,optimizer,20,logfile='log.txt',alert = True)
    except:
        sf.send_alert("Toxic Network training has failed")
        






























    
