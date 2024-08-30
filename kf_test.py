from KalmanFilter_PT import KalmanFilter
import numpy as np
import torch
import auto_LiRPA

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  #return tf.convert_to_tensor(np.array([x, y, s, r]).reshape((4, 1)), dtype=tf.float32)
  z = torch.nn.Parameter(torch.tensor([[x],[y],[s],[r]], requires_grad=True))
  return z.view(4, 1)

bbox = np.array([1689.   ,  385.   , 1835.62 ,  717.71 ,   67.567])

#define constant velocity model
kf = KalmanFilter(dim_x=7, dim_z=4) 
kf.model.F = torch.tensor(np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]))
kf.model.H = torch.tensor(np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]]))

kf.model.R[2:,2:] = kf.model.R[2:,2:] * 10.
kf.model.P[4:,4:] = kf.model.P[4:,4:] * 1000. #give high uncertainty to the unobservable initial velocities
kf.model.P *= 10.
#self.kf.Q[-1,-1].assign(self.kf.Q[-1,-1] * 0.01)
#self.kf.Q[4:,4:].assign(self.kf.Q[4:,4:] * 0.01)
kf.model.Q[2,2] = 50
kf.model.Q[-1,-1] = 50

kf.x.data[:4] = convert_bbox_to_z(bbox)
kf.model = auto_LiRPA.BoundedModule(kf.model, kf.x)