import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Net(torch.nn.Module):
    def __init__(self, hidden_size,num_layer,num_classes,embed_size,vocab_size,max_length,drop_prob=0.25):
        print("Net-> init")
        super(Net, self).__init__()
        
        #hidden_size =32
        #num_layer = 2
        #embeding_size= 100
        #n_vocab = 4500
        #max_length= 50
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_length = max_length
       
        self.cnn=nn.Conv1d(embed_size, hidden_size, 5)
        self.pooling= nn.AvgPool1d(2)
        
        self.cnn2 =nn.Conv1d(hidden_size, hidden_size, 4)
        self.pooling2 =nn.AvgPool1d(2)
        
        self.cnn3 =nn.Conv1d(hidden_size, hidden_size, 3)
        self.pooling3 =nn.AvgPool1d(2)
        
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.sigmoid=nn.Sigmoid()
        #self.lstm=nn.LSTM(embed_size, hidden_size, num_layer,batch_first=True,bidirectional=True)
        self.lstm=nn.LSTM(hidden_size, hidden_size, num_layer,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(2*hidden_size*35, num_classes) 
        self.relu=nn.ReLU()

    
    def init_hidden(self,batch_size):
        #print("model -> init_hidden")
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.num_layer*2, batch_size, self.hidden_size).zero_().cuda(),
#                       weight.new(self.num_layer*2, batch_size, self.hidden_size).zero_().cuda())
#         return hidden
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda()
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda()
        #print (h_0.shape)
        #print (c_0.shape)
        
        return (h_0, c_0)
   
    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        batch_size=x.size(0) 
        self.h_c = self.init_hidden(batch_size)
        
        x = self.embed(x)
  
        x = x.transpose(1,2)
        
        c = self.cnn(x)
        c = self.relu(c)
        p = self.pooling(c)

        
        c = self.cnn2(p)
        c = self.relu(c)
        x = self.pooling2(c)
        
        c = self.cnn3(x)
        c = self.relu(c)
        x = self.pooling3(c)

        x = p.transpose(1,2)
        x = F.tanh(x)
        #print(x.shape)
        #print(p.shape)
   
        output, self.h_c = self.lstm(x,self.h_c)
        #print(output.shape)
        h_t =output.contiguous()
        h_t= h_t.view(batch_size,-1)
        h_t= F.tanh(h_t)
        h_t = self.fc(h_t)
    
 

        return h_t
