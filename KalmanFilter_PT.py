import torch
import auto_LiRPA
from torch import nn, Tensor
import sys
from math import log
from numpy import isscalar
from copy import deepcopy
from filterpy.common import reshape_z
import numpy as np

class KalmanFilterModule(nn.Module):
    def __init__(self, dim_x, dim_z, dim_u=0):
        super(KalmanFilterModule, self).__init__()
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.P = torch.eye(dim_x)
        self.Q = torch.eye(dim_x)
        self.B = None
        self.F = torch.eye(dim_x)
        self.H = torch.zeros((dim_z, dim_x))
        self.R = torch.eye(dim_z)
        self._alpha_sq = 1.
        self.M = torch.zeros((dim_x, dim_z))
        self.z = torch.zeros(dim_z)

        self.K = torch.zeros((dim_x, dim_z))
        self.y = torch.zeros((dim_z, 1))
        self.S = torch.zeros((dim_z, dim_z))
        self.SI = torch.zeros((dim_z, dim_z))

        self._I = torch.eye(dim_x)
        self.P_prior = self.P.clone()

        self.P_post = self.P.clone()

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = torch.inverse

    def forward(self, x, z):
        return self.update(x, z)

    def predict(self, x, u=None, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F.to(torch.float32)
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = torch.matmul(torch.eye(self.dim_x), Q)

        if B is not None and u is not None:
            x = torch.matmul(F, x) + torch.matmul(B, u)
        else:
            #print(F)
            #print(self.x)
            x = torch.matmul(F, x)

        self.P = self._alpha_sq * torch.matmul(torch.matmul(F, self.P), torch.transpose(F, 0, 1)) + Q

        self.P_prior = self.P.detach().clone()

        return x

    def update(self, x, z):
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            #self.z = np.array([[None]*self.dim_z]).T
            self.z = torch.zeros(self.dim_z)
            self.x_post = x.detach().clone()
            self.P_post = self.P.detach().clone()
            self.y = torch.zeros((self.dim_z, 1))
            return
        
        #if R is None:
        R = self.R
        #elif isscalar(R):
            #R = torch.matmul(torch.eye(self.dim_z), R)

        #if H is None:
        z = torch.tensor(self._reshape_z(z, self.dim_z, self.dim_x))
        H = self.H.to(torch.float32)

        self.y = z - torch.matmul(H, x)

        PHT = torch.matmul(self.P, torch.transpose(H, 0, 1))

        self.S = torch.matmul(H, PHT) + R
        self.SI = self.inv(self.S)

        self.K = torch.matmul(PHT, self.SI)

        #assert(torch.all(z != x[:4]))
        #assert(torch.matmul(self.K, self.y).sum() != 0)

        x = x + torch.matmul(self.K, self.y)

        I_KH = self._I - torch.matmul(self.K, H)
        self.P = torch.matmul(torch.matmul(I_KH, self.P), torch.transpose(I_KH,0, 1)) + torch.matmul(torch.matmul(self.K, R), torch.transpose(self.K, 0, 1))

        self.z = deepcopy(z)
        self.x_post = x.detach().clone()
        self.P_post = self.P.detach().clone()

        return x
    
    def _reshape_z(self, z, dim_z, ndim):
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

class KalmanFilter():
    def __init__(self, dim_x, dim_z, dim_u=0):
        self.x = torch.zeros((dim_x, 1), requires_grad=True)
        self.x_prior = self.x.clone()
        self.x_post = self.x.clone()
        self.model = KalmanFilterModule(dim_x, dim_z, dim_u)
        self.lirpa_x = None
        self.lirpa_model = None
        self.lirpa_initialized = False
        self.eps = 0
        
    def initialize_lirpa(self, eps=0, device='cpu'):
        self.eps = eps
        self.lirpa_initialized = True
        ptb = auto_LiRPA.PerturbationLpNorm(eps=self.eps, norm=np.inf)
        self.lirpa_x = auto_LiRPA.BoundedTensor(self.x, ptb)
        self.lirpa_model = auto_LiRPA.BoundedModule(self.model,\
            (torch.zeros(self.x.shape), torch.zeros((self.model.dim_z, 1))),\
            bound_opts={'conv_mode': 'matrix'}, device=device)

    def predict(self):
        self.x = self.model.predict(self.x)
        self.x_prior = self.x.clone()

    def update(self, z):
        self.x = self.model(self.x, z)
        pass

    def compute_prev_bounds(self, z):
        if not self.lirpa_initialized:
            print("Lirpa not initialized to most recent state.")
            return
        ptb = auto_LiRPA.PerturbationLpNorm(eps=self.eps, norm=np.inf)
        z = auto_LiRPA.BoundedTensor(z, ptb)
        self.lirpa_x = self.lirpa_model(self.lirpa_x, z)
        lb, ub = self.lirpa_model.compute_bounds(method='ibp')
        self.lirpa_initialized = False
        print(ub)
        print(lb)

    def __str__(self):
        return f'P: {self.model.P}\n' +\
        f'Q: {self.model.Q}\n' +\
        f'B: {self.model.B}\n' +\
        f'F: {self.model.F}\n' +\
        f'H: {self.model.H}\n' +\
        f'R: {self.model.R}\n' +\
        f'M: {self.model.M}\n' +\
        f'z: {self.model.z}\n' +\
        f'K: {self.model.K}\n' +\
        f'y: {self.model.y}\n' +\
        f'S: {self.model.S}\n' +\
        f'SI: {self.model.SI}\n' +\
        f'_I: {self.model._I}\n' +\
        f'P_prior: {self.model.P_prior}\n' +\
        f'P_post: {self.model.P_post}\n' +\
        f'x: {self.x}'
