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
        return x, P

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
        self._I = torch.eye(dim_x, dtype=torch.float32)
        self.inv = torch.inverse

    def forward(self, x, z, P):
        y = z - torch.matmul(self.H, x)

        PHT = torch.matmul(P, torch.transpose(self.H, 0, 1))

        S = torch.matmul(self.H, PHT) + self.R
        SI = self.inv(S)

        K = torch.matmul(PHT, SI)

        x = x + torch.matmul(K, y)

        I_KH = self._I - torch.matmul(K, self.H)
        P = torch.matmul(torch.matmul(I_KH, P), torch.transpose(I_KH, 0, 1)) + torch.matmul(torch.matmul(K, self.R), torch.transpose(K, 0, 1))

        z = z.detach().clone()

        return x, z, P

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

        self.x = torch.zeros((dim_x, 1), requires_grad=True, dtype=torch.float32)
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
        self.x, self.P = self.predict_module(self.x, self.P)
        return self.x
    
    def update(self, z):
        self.x, self.z, self.P = self.update_module(self.x, z, self.P)
        return self.x
    
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