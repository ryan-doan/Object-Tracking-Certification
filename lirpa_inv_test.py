import torch
from torch import nn, Tensor
import auto_LiRPA
import numpy as np

class InverseModule(nn.Module):
    def __init__(self):
        super(InverseModule, self).__init__()
        self.P = torch.tensor([[0, 1],[1, 0]], dtype=torch.float32)
        self.flip_sign = torch.tensor([[1,0],[0,-1]], dtype=torch.float32)

    def forward(self, x):
        det = x[0][0]*x[1][1] - x[0][1]*x[1][0]
        # This transform the matrix from [[a, b], [c, d]] to [[d, b], [c, a]]
        # I've isolate the errors to the statement self.P.matmul(x)
        x = self.P.matmul(x).matmul(self.P)
        # Flip b and c sign
        x = x.matmul(self.flip_sign)
        return (1/det) * x
    
if __name__ == "__main__":
    x = Tensor([[1, 2], [3, 4]])
    x.requires_grad_()
    model = InverseModule()
    #model.forward(x)
    lirpa_model = auto_LiRPA.BoundedModule(model, x)
    ptb = auto_LiRPA.PerturbationLpNorm(eps=0.1, norm = np.inf)
    x = auto_LiRPA.BoundedTensor(x, ptb)
    
    y = lirpa_model(x)
    lb, ub = lirpa_model.compute_bounds()
    print(lb)
    print(ub)
    pass