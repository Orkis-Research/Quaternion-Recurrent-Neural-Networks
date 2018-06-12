import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch.optim
from torch.autograd import Variable
from quaternion_models import QuaternionLinearAutograd, QuaternionLinear

class QRNN(nn.Module):
    def __init__(self, feat_size, hidden_size, nb_hidden):
        super(QRNN, self).__init__()
        
        # Reading options:
        self.input_dim=feat_size
        self.hidden_dim=hidden_size
        self.N_hid=nb_hidden
        self.num_classes=feat_size
    
        # List initialization
        self.wx  = nn.ModuleList([]) 
        self.uh  = nn.ModuleList([]) 
        
        # Activation        
        self.act=nn.Sigmoid()

        curr_dim=self.input_dim
        
        for i in range(self.N_hid):
                      
          self.wx.append(nn.Linear(curr_dim, self.hidden_dim))
           
          # uh initialization
          self.uh.append(QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim))
          curr_dim=self.hidden_dim   
        
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.0002)
    
    def forward(self, x):
    

        h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))   
        x      = x.cuda()
        h_init = h_init.cuda()

            
        wx_out=self.wx[0](x)
        hiddens = []
        pre_act = []
        h=h_init
        
        for k in range(x.shape[0]):
            at=wx_out[k]+self.uh[0](h)
            h=at

        # Delimiters, time to generate !
        out = []
        hiddens = []
        pre_act = []
        
        for k in range(x.shape[0]):
            at=self.uh[0](h)
            h=at
            output = self.act(self.fco(h))
            out.append(output.unsqueeze(0))
 
        return torch.cat(out,0)

class RNN(nn.Module):
    def __init__(self, feat_size, hidden_size, nb_hidden):
        super(RNN, self).__init__()
        
        # Reading options:
        self.input_dim=feat_size
        self.hidden_dim=hidden_size
        self.N_hid=nb_hidden
        self.num_classes=feat_size
       
        # List initialization
        self.wx  = nn.ModuleList([]) 
        self.uh  = nn.ModuleList([]) 
        
        # Activation  
        self.act=nn.Sigmoid()

        curr_dim=self.input_dim
        
        for i in range(self.N_hid):
                      
          self.wx.append(nn.Linear(curr_dim, self.hidden_dim))
           
          # uh initialization
          self.uh.append(nn.Linear(self.hidden_dim, self.hidden_dim))
          curr_dim=self.hidden_dim   
        
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.0002)
    
    def forward(self, x):
    

        h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))   
        x      = x.cuda()
        h_init = h_init.cuda()

            
        wx_out=self.wx[0](x)

        hiddens = []
        pre_act = []
        h=h_init
        
        for k in range(x.shape[0]):
            at=wx_out[k]+self.uh[0](h)
            h=at

        # Delimiters, time to generate !
        out = []
        hiddens = []
        pre_act = []
        
        for k in range(x.shape[0]):
            at=self.uh[0](h)
            h=at
            output = self.act(self.fco(h))

            out.append(output.unsqueeze(0))
        
        return torch.cat(out,0)
