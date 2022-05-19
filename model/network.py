import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MLP(nn.Module): 
    def __init__(self, input_size, output_size): 
       super(MLP, self).__init__() 
        
       self.layer1 = nn.Linear(input_size, 512) 
       self.layer2 = nn.Linear(512, 256) 
       self.layer3 = nn.Linear(256, output_size) 

    def forward(self, x): 
        x1 = F.relu(self.layer1(x)) 
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2) 
        return x3 