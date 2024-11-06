from KalmanFilter_PT import KalmanFilter
import numpy as np
import torch
import auto_LiRPA
import math

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

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = torch.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    #return tf.convert_to_tensor(np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4)), dtype=tf.float32)
    return torch.tensor([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape(1,4)
  else:
    return torch.tensor([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def generate_data(start, frames, x_delta_mean, x_std_dev, y_delta_mean, y_std_dev):
  curr_frame = start #
  data = []
  label = []
  for i in range(frames):
      np.random.seed(42)
      x_delta = np.random.normal(x_delta_mean, x_std_dev)
      y_delta = np.random.normal(y_delta_mean, y_std_dev)
      data.append(torch.tensor(curr_frame, requires_grad=True).reshape(4, 1))
      curr_frame[0] += x_delta
      curr_frame[1] += y_delta
      label.append(torch.tensor(curr_frame).reshape(4, 1))

  return data, label

def compute_iou(output, label):
  x1 = torch.max(output[:, 0], label[:, 0])
  y1 = torch.max(output[:, 1], label[:, 1])
  x2 = torch.min(output[:, 2], label[:, 2])
  y2 = torch.min(output[:, 3], label[:, 3])

  inter_area = (x2-x1) * (y2-y1)

  output_area = (output[:, 2] - output[:, 0]) * (output[:, 3] - output[:, 1])
  label_area = (label[:, 2] - label[:, 0]) * (label[:, 3] - label[:, 1])

  union_area = output_area + label_area - inter_area

  iou = torch.where(union_area > 0, inter_area / union_area, torch.tensor(0.0))

  return iou

def compute_l2_dist(output, label):
  return math.sqrt(math.pow(output[0] - label[0], 2) + math.pow(output[1] - label[1], 2))

if __name__ == '__main__':
    device = torch.device('cpu')
    #generate data set
    x_delta_mean = 20
    x_std_dev = 4
    y_delta_mean = 0
    y_std_dev = 2
    data, label = generate_data([1762.3, 551.36, 48782, 0.44068], 10, x_delta_mean, x_std_dev, y_delta_mean, y_std_dev)

    def kf_initialize():
      #define constant velocity model
      kf = KalmanFilter(dim_x=7, dim_z=4) 
      kf.predict_module.F = torch.tensor([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],\
                                      [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], dtype=torch.float32)
      kf.update_module.H = torch.tensor([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=torch.float32)

      kf.update_module.R[2:,2:] = kf.update_module.R[2:,2:] * 10.
      kf.P[:, 4:,4:] = kf.P[:, 4:,4:] * 1000. #give high uncertainty to the unobservable initial velocities
      kf.P *= 10.
      #self.kf.Q[-1,-1].assign(self.kf.Q[-1,-1] * 0.01)
      #self.kf.Q[4:,4:].assign(self.kf.Q[4:,4:] * 0.01)
      kf.predict_module.Q[2,2] = 50
      kf.predict_module.Q[-1,-1] = 50
      #self.kf.model.Q[-1,-1] *= 0.01
      #self.kf.model.Q[4:,4:] *= 0.01
      kf.x.data[:, :4] = data[0]
      #kf.initialize_lirpa()
      #label[0][:2] += 10
      return kf

    kf_lirpa = kf_initialize()
    kf_lirpa.initialize_lirpa()
    kf_u = kf_initialize()
    kf_l = kf_initialize()

    #predict
    total_dist = 0
    for i in range(len(label)):
      kf_l.predict()
      kf_u.predict()
      kf_lirpa.predict()

      kf_lirpa.update(label[i].reshape((1, 4, 1)))
      if i == 0:
        label[0][:2] -= 10
      kf_l.update(label[i].reshape((1, 4, 1)))
      if i == 0:
        label[0][:2] += 20
      kf_u.update(label[i].reshape((1, 4, 1)))
      pass
    #kf.compute_prev_bounds_update()

    #print(f'Average distance between prediction and label: {total_dist/len(label)}')