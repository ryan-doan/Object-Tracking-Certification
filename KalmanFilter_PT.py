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

class KalmanFilterPredict(nn.Module):
    def __init__(self, dim_x):
        super(KalmanFilterPredict, self).__init__()
        self._alpha_sq = 1.
        self.Q = torch.eye(dim_x, dtype=torch.float32)
        self.F = torch.eye(dim_x, dtype=torch.float32)
        #self.Fx = nn.Linear(dim_x, dim_x, bias=False)

    def forward(self, x, P):
        #x = self.Fx(x)
        x = torch.matmul(x.transpose(1, 2), self.F.transpose(0, 1)).transpose(1, 2)
        FP = torch.matmul(P.transpose(1, 2), self.F.transpose(0, 1)).transpose(1, 2)
        P = self._alpha_sq * torch.matmul(FP, self.F.transpose(0, 1)) + self.Q
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
        self.R = torch.eye(dim_z, dtype=torch.float32)
        self.H = torch.zeros((dim_z, dim_x), dtype=torch.float32)
        self._Ix = torch.eye(dim_x, dtype=torch.float32)
        self._Iz = torch.eye(dim_z, dtype=torch.float32)
        self._zero_diagonal = torch.ones(dim_z, dim_z).fill_diagonal_(0)
        #self.z_out = 
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
        PHT = torch.matmul(P, self.H.transpose(0, 1))

        S = torch.matmul(PHT.transpose(1, 2), self.H.transpose(0, 1)).transpose(1, 2) + self.R
        #SI = self.inv(S)
        # IMPORTANT! THIS INVERSE METHOD ONLY WORKS WITH DIAGONAL MATRICES (R and H must be diagonal)
        SI = 1 / (self._zero_diagonal + S) - self._zero_diagonal
        #SI = self.inv(S)

        # K contains information on our vector field.
        K = torch.matmul(PHT, SI)

        # y contains information on x, y coordinates of our field.
        y = z - torch.matmul(x.transpose(1, 2), self.H.transpose(0, 1)).transpose(1, 2)

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
    def __init__(self, dim_x, dim_z, id=0, dim_u=0, method='reset-velo'):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.id = id
        self.lirpa_initialized = False
        self.method = method

        self.x = torch.zeros((1, dim_x, 1), dtype=torch.float32)
        self.P = torch.eye(dim_x, dtype=torch.float32).unsqueeze(0)
        self.z = torch.zeros((1, dim_z, 1), dtype=torch.float32)
        
        self.x_l = None
        self.x_u = None
        self.x_l_u = None
        self.x_u_l = None

        self.predict_module = KalmanFilterPredict(dim_x)
        self.update_module = KalmanFilterUpdate(dim_x, dim_z)

        self.countdown = 10
        self.perturbed = True

    def predict(self):
        def reset_velo():
            if self.x_l != None:
                ptb = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=self.x_l, x_U=self.x_u)
                self.x = auto_LiRPA.BoundedTensor(self.x, ptb)
                
            out = self.predict_module(self.x, self.P)
            self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
            self.P = out[:, 1:]
            self._compute_prev_bounds_predict()
            return self.x[0]

        def split_ptb():
            if self.x_l == None:
                # Same calculation if x_l is None
                return reset_velo()
            
            # Compute lower half of ptb set
            ptb_x_l = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L = self.x_l, x_U = self.x_l_u)
            x = auto_LiRPA.BoundedTensor(self.x, ptb_x_l)
            out = self.predict_module(x, self.P)
            self._compute_prev_bounds_predict()

            # Compute upper half of ptb set
            ptb_x_l = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L = self.x_u_l, x_U = self.x_u)
            x = auto_LiRPA.BoundedTensor(self.x, ptb_x_l)
            out = self.predict_module(x, self.P)
            self._compute_prev_bounds_predict()
            self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
            self.P = out[:, 1:]
            return self.x[0]

        if self.lirpa_initialized:
            if self.method == 'reset-velo':
                return reset_velo()
            elif self.method == 'split-ptb':
                return split_ptb()
            else:
                raise NotImplementedError
        
        out = self.predict_module(self.x, self.P)
        self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
        self.P = out[:, 1:]
        if self.lirpa_initialized:
            self._compute_prev_bounds_predict()
        return self.x[0]
    
    def update(self, z):
        def reset_velo(z):
            self.x[:, 4:6] = 0
            self.x_l[:, 4:6] = 0
            self.x_u[:, 4:6] = 0
            z_l = z.detach().clone()
            z_u = z.detach().clone()
            z_l[:, :2] -= self.initial_eps
            z_u[:, :2] += self.initial_eps
            ptb_z = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=z_l, x_U=z_u)
            z = auto_LiRPA.BoundedTensor(z, ptb_z)
            if self.x_l != None:
                ptb = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=self.x_l, x_U=self.x_u)
                self.x = auto_LiRPA.BoundedTensor(self.x, ptb)
            out = self.update_module(self.x, z, self.P)
            self._compute_prev_bounds_update()
            self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
            self.P = out[:, 1:]
            return self.x[0]
        
        def split_ptb(z):
            # Compute lower half of ptb set
            z_l = z.detach().clone()
            z_l[:, :2] -= self.initial_eps
            ptb_z_l = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=z_l, x_U=z)
            z = auto_LiRPA.BoundedTensor(z, ptb_z_l)
            x = self.x
            if self.x_l != None:
                ptb_x = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=self.x_l, x_U=self.x_l_u)
                x = auto_LiRPA.BoundedTensor(self.x, ptb_x)
            out = self.update_module(x, auto_LiRPA.BoundedTensor(z, ptb_z_l), self.P)
            self._compute_prev_bounds_update(l=True)

            # Compute upper half of ptb set
            z_u = z.detach().clone()
            z_u[:, :2] += self.initial_eps
            if self.x_u != None:
                ptb_x = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=self.x_u_l, x_U=self.x_u)
                x = auto_LiRPA.BoundedTensor(self.x, ptb_x)
            ptb_z_u = auto_LiRPA.PerturbationLpNorm(norm=np.inf, x_L=z, x_U=z_u)
            z = auto_LiRPA.BoundedTensor(z, ptb_z_l)
            out = self.update_module(x, auto_LiRPA.BoundedTensor(z, ptb_z_u), self.P)
            self._compute_prev_bounds_update(u=True)

            #Return results
            self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
            self.P = out[:, 1:]
            return self.x[0]
        
        if self.lirpa_initialized:
            if self.method == 'reset-velo':
                return reset_velo(z)
            elif self.method == 'split-ptb':
                return split_ptb(z)
            else:
                raise NotImplementedError
        
        #self.x, self.z, self.P = self.update_module(self.x, z, self.P)
        out = self.update_module(self.x, z, self.P)
        self.x = torch.reshape(out[:, 0], (1, self.dim_x,1))
        self.P = out[:, 1:]
        return self.x[0]

    def initialize_lirpa(self, eps = 10):
        self.x.requires_grad_()
        self.P.requires_grad_()
        self.z.requires_grad_()
        self.lirpa_initialized = True
        self.initial_eps = eps
        self.predict_module = auto_LiRPA.BoundedModule(self.predict_module, \
                                                       global_input=(self.x, self.P),\
                                                        device="cpu")
        self.update_module = auto_LiRPA.BoundedModule(self.update_module, \
                                                      global_input=(self.x, self.z, self.P),\
                                                        device="cpu")

    def _compute_prev_bounds_predict(self, method='crown', l=False, u=False):
        lb, ub = self.predict_module.compute_bounds(method=method)
        #print(f'Lower bound: {lb[:, 0, :4]}')
        #print(f'Upper bound: {ub[:, 0, :4]}')
        if not l and not u:
            self.x_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))
            self.x_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
            return
        if l:
            self.x_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))
            self.x_l_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
        if u:
            self.x_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
            self.x_u_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))

    def _compute_prev_bounds_update(self, method='crown', l=False, u=False):
        lb, ub = self.update_module.compute_bounds(method=method)
        #print(f'Lower bound: {lb[:, 0, :4]}')
        #print(f'Upper bound: {ub[:, 0, :4]}')
        if not l and not u:
            self.x_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))
            self.x_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
            return
        if l:
            self.x_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))
            self.x_l_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
        if u:
            self.x_u = torch.reshape(ub[:, 0], (1, self.dim_x,1))
            self.x_u_l = torch.reshape(lb[:, 0], (1, self.dim_x,1))
    
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