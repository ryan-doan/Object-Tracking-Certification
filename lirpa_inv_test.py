import torch
from torch import nn, Tensor
import auto_LiRPA
import numpy as np

class InverseModule(nn.Module):
    def __init__(self):
        super(InverseModule, self).__init__()

    def forward(self, x):
        return self._blockwise_inversion(x)
    
    def _blockwise_inversion(self, x):
        a = x[:, :2, :2]
        b = x[:, :2, 2:]
        c = x[:, 2:, :2]
        d = x[:, 2:, 2:]

        a_inv = self._two_by_two_inverse(a)

        schur_comp = self._two_by_two_inverse(d-c.matmul(a_inv).matmul(b))

        A = a_inv+a_inv.matmul(b).matmul(schur_comp).matmul(c).matmul(a_inv)
        B = -a_inv.matmul(b).matmul(schur_comp)
        C = -schur_comp.matmul(c).matmul(a_inv)
        D = schur_comp

        return torch.cat((torch.cat((A, B), 2), torch.cat((C, D), 2)), 1)
    
    def _two_by_two_inverse(self, x):
        a = x[:, :1, :1]
        b = x[:, :1, 1:]
        c = x[:, 1:, :1]
        d = x[:, 1:, 1:]
        det = (a * d - b * c)
        #x = self.P.matmul(x).matmul(self.P)
        #x = x.matmul(self.P).transpose(-1, -2).matmul(self.P)
        flipped_matrix = torch.cat((torch.cat((d, -b), 2), torch.cat((-c, a), 2)), 1)

        return (1/det) * flipped_matrix
    
class TwoByTwoInverseModule(nn.Module):
    def __init__(self):
        super(TwoByTwoInverseModule, self).__init__()

    def forward(self, x):
        a = x[:, :1, :1]
        b = x[:, :1, 1:]
        c = x[:, 1:, :1]
        d = x[:, 1:, 1:]
        det = (a * d - b * c)
        #x = self.P.matmul(x).matmul(self.P)
        #x = x.matmul(self.P).transpose(-1, -2).matmul(self.P)
        flipped_matrix = torch.cat((torch.cat((d, -b), 2), torch.cat((-c, a), 2)), 1)

        return (1/det) * flipped_matrix
    
if __name__ == "__main__":
    x = Tensor([[[10, 2],
                [3, 4]]])
    model = TwoByTwoInverseModule()
    lirpa_model = auto_LiRPA.BoundedModule(model, Tensor([[[1, 0], [0, 1]]]))
    ptb = auto_LiRPA.PerturbationLpNorm(eps=0.1, norm = np.inf)
    x = auto_LiRPA.BoundedTensor(x, ptb)
    
    y = lirpa_model(x)
    y_true = torch.inverse(x)
    lb, ub = lirpa_model.compute_bounds()
    print(lb)
    print(ub)

    x = Tensor([[[10.0, 0.0, 0.0, 0.0],
                    [0.0, 60.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 7.0]]])
    x.requires_grad_()
    model = InverseModule()
    #model.forward(x)
    lirpa_model = auto_LiRPA.BoundedModule(model, 
                Tensor([[[1.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0]]]))
    ptb = auto_LiRPA.PerturbationLpNorm(eps=0.1, norm = np.inf)
    x = auto_LiRPA.BoundedTensor(x, ptb)
    
    y = lirpa_model(x)
    y_true = torch.inverse(x)
    lb, ub = lirpa_model.compute_bounds()
    print(lb)
    print(ub)
    pass