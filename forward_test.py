import torch.nn as nn
from torch import Tensor
import auto_LiRPA
import numpy as np

class ForwardModule(nn.Module):
    def __init__(self):
        super(ForwardModule, self).__init__()
        self.w = Tensor([[[7, 8],[9, 10]]])
        #ptb = auto_LiRPA.PerturbationLpNorm(eps=1, norm=np.inf)
        #self.w = 

    def forward(self, x):
        return self.w.matmul(x)
    
class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.w = Tensor([[5, 6],[7, 8]])
        #ptb = auto_LiRPA.PerturbationLpNorm(eps=1, norm=np.inf)
        #self.w = 

    def forward(self, x):
        return x.matmul(self.w)
    
if __name__ == "__main__":
    model = ForwardModule()
    #model2 = LinearModule()
    x = Tensor([[5, 6], [7, 18]])
    model = auto_LiRPA.BoundedModule(model, x)
    #model2 = auto_LiRPA.BoundedModule(model2, x.unsqueeze(0))
    ptb = auto_LiRPA.PerturbationLpNorm(eps=1, norm=np.inf)
    x = auto_LiRPA.BoundedParameter(x, ptb)
    y = model(x)
    #y2 = model2(auto_LiRPA.BoundedTensor(x.unsqueeze(0), ptb))
    #model2.compute_bounds(method='backward')
    model.compute_bounds(method='backward')