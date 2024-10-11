import torch
import auto_LiRPA
from torch import nn, Tensor
import sys
from math import log
from numpy import isscalar
from copy import deepcopy
from filterpy.common import reshape_z
import numpy as np

class KalmanFilterPredict(nn.Module):
    def __init__(self, dim_x):
        super(KalmanFilterPredict, self).__init__()
        self._alpha_sq = 1.
        self.Q = torch.eye(dim_x, dtype=torch.float32)
        self.F = torch.eye(dim_x, dtype=torch.float32)
        #self.Fx = nn.Linear(dim_x, dim_x, bias=False)

    def forward(self, x, P):
        #x = self.Fx(x)
        x = torch.matmul(self.F, x)
        P = self._alpha_sq * torch.matmul(torch.matmul(self.F, P), torch.transpose(self.F, 0, 1)) + self.Q
        return torch.cat((x.T, P))

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
        self.R = torch.eye(dim_z, dtype=torch.float32)
        self.H = torch.zeros((dim_z, dim_x), dtype=torch.float32)
        self._Ix = torch.eye(dim_x, dtype=torch.float32)
        self._Iz = torch.eye(dim_z, dtype=torch.float32)
        self.inv = torch.inverse
        self.level_of_approximation = 50

    def forward(self, x, z, P):
        y = z - torch.matmul(self.H, x)

        PHT = torch.matmul(P, torch.transpose(self.H, 0, 1))

        S = torch.matmul(self.H, PHT) + self.R
        SI = self.inv(S)

        K = torch.matmul(PHT, SI)

        x = x + torch.matmul(K, y)

        I_KH = self._Ix - torch.matmul(K, self.H)
        P = torch.matmul(torch.matmul(I_KH, P), torch.transpose(I_KH, 0, 1)) + torch.matmul(torch.matmul(K, self.R), torch.transpose(K, 0, 1))

        #z = z.detach().clone()

        return torch.cat((x.T, P))
    
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

        self.x = torch.zeros((dim_x, 1), dtype=torch.float32)
        self.P = torch.eye(dim_x, dtype=torch.float32)
        self.z = torch.zeros(dim_z, dtype=torch.float32)
        #self.Q = torch.eye(dim_x)
        #self.B = None
        #self.F = torch.eye(dim_x)
        #self.H = torch.zeros((dim_z, dim_x))
        #self.R = torch.eye(dim_z)
        #self._alpha_sq = 1.
        #self.M = torch.zeros((dim_x, dim_z))
        

        #self.K = torch.zeros((dim_x, dim_z))
        #self.y = torch.zeros((dim_z, 1))
        #self.S = torch.zeros((dim_z, dim_z))
        #self.SI = torch.zeros((dim_z, dim_z))

        #self._I = torch.eye(dim_x)
        #self.P_prior = self.P.clone()

        #self.P_post = self.P.clone()

        #self._log_likelihood = log(sys.float_info.min)
        #self._likelihood = sys.float_info.min
        #self._mahalanobis = None

        #self.inv = torch.inverse

        self.predict_module = KalmanFilterPredict(dim_x)
        self.update_module = KalmanFilterUpdate(dim_x, dim_z)

    def predict(self):
        zero_ptb = auto_LiRPA.PerturbationLpNorm(0, np.inf)
        self.x = auto_LiRPA.BoundedTensor(self.x, zero_ptb)
        self.P = auto_LiRPA.BoundedTensor(self.P, zero_ptb)
        #self.x, self.P = self.predict_module(self.x, self.P)
        out = self.predict_module(self.x, self.P)
        self.x = torch.reshape(out[0], (self.dim_x,1))
        self.P = out[1:]
        return self.x
    
    def update(self, z):
        ptb = auto_LiRPA.PerturbationLpNorm(0.1, np.inf)
        zero_ptb = auto_LiRPA.PerturbationLpNorm(0, np.inf)
        self.x = auto_LiRPA.BoundedTensor(self.x, zero_ptb)
        z = auto_LiRPA.BoundedTensor(z, ptb)
        self.P = auto_LiRPA.BoundedTensor(self.P, zero_ptb)
        #self.x, self.z, self.P = self.update_module(self.x, z, self.P)
        out = self.update_module(self.x, z, self.P)
        self.x = torch.reshape(out[0], (self.dim_x,1))
        self.P = out[1:]
        return self.x

    def initialize_lirpa(self):
        self.x.requires_grad_()
        self.z.requires_grad_()
        self.P.requires_grad_()
        self.lirpa_initialized = True
        self.predict_module = auto_LiRPA.BoundedModule(self.predict_module, \
                                                       global_input=(self.x, self.P),\
                                                        device="cpu")
        self.update_module = auto_LiRPA.BoundedModule(self.update_module, \
                                                      global_input=(self.x, self.z, self.P),\
                                                        device="cpu")

    def compute_prev_bounds_update(self):
        if not self.lirpa_initialized:
            print('Lirpa not initialized')
            return
        lb, ub = self.update_module.compute_bounds()
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