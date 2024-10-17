import torch
import auto_LiRPA
from torch import nn, Tensor
import sys
from math import log
from numpy import isscalar
from copy import deepcopy
from filterpy.common import reshape_z
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

class KalmanFilterPredict(nn.Module):
    def __init__(self, dim_x):
        super(KalmanFilterPredict, self).__init__()
        self._alpha_sq = 1.
        self.Q = torch.eye(dim_x, dtype=torch.float32).unsqueeze(0)
        self.F = torch.eye(dim_x, dtype=torch.float32).unsqueeze(0)
        #self.Fx = nn.Linear(dim_x, dim_x, bias=False)

    def forward(self, x, P):
        #x = self.Fx(x)
        x = torch.matmul(self.F, x)
        P = self._alpha_sq * torch.matmul(torch.matmul(self.F, P), torch.transpose(self.F, 1, 2)) + self.Q
        return torch.cat((x.transpose(1, 2), P), dim=1)

    '''
    def initialize_params(self, F=None):
        with torch.no_grad():
            if F is not None:
                FT = F.T
                self.Fx.weight[:,:] = FT[:,:]
                pass
    '''

class KalmanFilterUpdate(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(KalmanFilterUpdate, self).__init__()
        self.R = torch.eye(dim_z, dtype=torch.float32).unsqueeze(0)
        self.H = torch.zeros((dim_z, dim_x), dtype=torch.float32).unsqueeze(0)
        self._Ix = torch.eye(dim_x, dtype=torch.float32).unsqueeze(0)
        self._Iz = torch.eye(dim_z, dtype=torch.float32).unsqueeze(0)
        self._zero_diagonal = torch.ones(dim_z, dim_z).fill_diagonal_(0)
        self.inv = InverseModule()
        self.level_of_approximation = 50

    def _is_diagonal(self, tensor):
        # Check if the tensor is square
        if tensor.shape[2] != tensor.shape[1]:
            return False
        
        # Create a boolean mask for the diagonal elements
        diagonal_mask = torch.eye(tensor.shape[1], dtype=bool).unsqueeze(0)
        
        # Check if all non-diagonal elements are zero
        return torch.all(tensor[~diagonal_mask] == 0)

    def forward(self, x, z, P):
        PHT = torch.matmul(P, torch.transpose(self.H, 1, 2))

        S = torch.matmul(self.H, PHT) + self.R
        #SI = self.inv(S)
        # IMPORTANT! THIS IS JUST A WORKAROUND CURRENTLY! THIS INVERSE METHOD ONLY WORKS WITH DIAGONAL MATRICES (R and H must be diagonal)
        #SI = 1 / (self._zero_diagonal + S) - self._zero_diagonal
        SI = self.inv(S)

        K = torch.matmul(PHT, SI)

        y = z - torch.matmul(self.H, x)

        x = x + torch.matmul(K, y)

        I_KH = self._Ix - torch.matmul(K, self.H)
        P = torch.matmul(torch.matmul(I_KH, P), torch.transpose(I_KH, 1, 2))\
              + torch.matmul(torch.matmul(K, self.R), torch.transpose(K, 1, 2))

        #z = z.detach().clone()

        return torch.cat((x.transpose(1, 2), P), dim=1)
    
    def _neumann_inverse_method(self, S):
        a = 0.001
        SI = self._Iz

        for i in range(1, self.level_of_approximation):
            T = self._Iz - (S * a)
            Spow = T
            for j in range(i-1):
                Spow = torch.matmul(Spow, T)
            SI = SI + Spow

        return SI * a

class KalmanFilter():
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.lirpa_initialized = False

        self.x = torch.zeros((1, dim_x, 1), dtype=torch.float32)
        self.P = torch.tensor(torch.eye(dim_x, dtype=torch.float32)).unsqueeze(0)
        self.z = torch.zeros((1, dim_z, 1), dtype=torch.float32)
        
        self.x_l = None
        self.x_u = None
        self.P_l = None
        self.P_u = None

        self.predict_module = KalmanFilterPredict(dim_x)
        self.update_module = KalmanFilterUpdate(dim_x, dim_z)

    def predict(self):
        ptb = auto_LiRPA.PerturbationLpNorm(0.1, np.inf)
        zero_ptb = auto_LiRPA.PerturbationLpNorm(0, np.inf)
        self.x = auto_LiRPA.BoundedTensor(self.x, ptb)
        self.P = auto_LiRPA.BoundedTensor(self.P, zero_ptb)
        #self.x, self.P = self.predict_module(self.x, self.P)
        out = self.predict_module(self.x, self.P)
        self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
        self.P = out[:, 1:]
        return self.x[0]
    
    def update(self, z):
        ptb_z = auto_LiRPA.PerturbationLpNorm(self.initial_eps, np.inf)
        z = auto_LiRPA.BoundedTensor(z, ptb_z)
        if self.x_L == None:
            zero_ptb = auto_LiRPA.PerturbationLpNorm(0, np.inf)
            self.x = auto_LiRPA.BoundedTensor(self.x, zero_ptb)
            self.P = auto_LiRPA.BoundedTensor(self.P, zero_ptb)
        else:
            ptb_x = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L = self.x_l, x_U = self.x_u)
            ptb_P = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L = self.P_l, x_U = self.P_u)
            self.x = auto_LiRPA.BoundedTensor(self.x, ptb_x)
            self.P = auto_LiRPA.BoundedTensor(self.P, ptb_P)
        
        #self.x, self.z, self.P = self.update_module(self.x, z, self.P)
        out = self.update_module(self.x, z, self.P)
        self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
        self.P = out[:, 1:]
        return self.x[0]

    def initialize_lirpa(self, eps = 0.1):
        self.x.requires_grad_()
        self.z.requires_grad_()
        self.P.requires_grad_()
        self.lirpa_initialized = True
        self.initial_eps = eps
        self.predict_module = auto_LiRPA.BoundedModule(self.predict_module, \
                                                       global_input=(self.x, self.P),\
                                                        device="cpu")
        self.update_module = auto_LiRPA.BoundedModule(self.update_module, \
                                                      global_input=(self.x, self.z, self.P),\
                                                        device="cpu")

    def compute_prev_bounds_predict(self):
        if not self.lirpa_initialized:
            print('Lirpa not initialized')
            return
        lb, ub = self.predict_module.compute_bounds()
        print(ub)
        print(lb)

    def compute_prev_bounds_update(self):
        if not self.lirpa_initialized:
            print('Lirpa not initialized')
            return
        lb, ub = self.update_module.compute_bounds(method='backward')
        print(ub)
        print(lb)
    
    def _reshape_z(self, z, dim_z, ndim):
        with torch.no_grad:
            z = torch.atleast_2d(z)

            if z.shape[1] == dim_z:
                z = z.T

            if z.shape != (dim_z, 1):
                raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

            if ndim == 1:
                z = z[:, 0]

            if ndim == 0:
                z = z[0, 0]

            return z