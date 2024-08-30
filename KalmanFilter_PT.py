import torch
from torch import nn, Tensor
import sys
from math import log
from numpy import isscalar
from copy import deepcopy
from filterpy.common import reshape_z

class KalmanFilter(nn.Module):
    def __init__(self, dim_x, dim_z, dim_u=0):
        super(KalmanFilter, self).__init__()
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.zeros((dim_x, 1))
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
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

        self.x_post = self.x.clone()

        self.P_post = self.P.clone()

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = torch.inverse

    def forward(self):
        return self.predict()

    def predict(self, u=None, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F.to(torch.float32)
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = torch.eye(self.dim_x) * Q

        if B is not None and u is not None:
            self.x = torch.matmul(F, self.x) + torch.matmul(B, u)
        else:
            #print(F)
            #print(self.x)
            self.x = torch.matmul(F, self.x)

        self.P = self._alpha_sq * torch.matmul(torch.matmul(F, self.P), torch.transpose(F, 0, 1)) + Q

        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

    def update(self, z, R=None, H=None):
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            #self.z = np.array([[None]*self.dim_z]).T
            self.z = torch.zeros(self.dim_z)
            self.x_post = self.x.clone()
            self.P_post = self.P.clone()
            self.y = torch.zeros((self.dim_z, 1))
            return
        
        if R is None:
            R = self.R
        elif isscalar(R):
            R = torch.eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.dim_x)
            H = self.H.to(torch.float32)

        self.y = z - torch.matmul(H, self.x)

        PHT = torch.matmul(self.P, torch.transpose(H))

        self.S = torch.matmul(H, PHT) + R
        self.SI = self.inv(self.S)

        self.K = torch.matmul(PHT, self.SI)

        self.x = self.x + torch.matmul(self.K, self.y)


        I_KH = self._I - torch.matmul(self.K, H)
        self.P = torch.matmul(torch.matmul(I_KH, self.P), torch.transpose(I_KH)) + torch.matmul(torch.matmul(self.K, R), torch.transpose(self.K))

        self.z = deepcopy(z)
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()