import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Bounds visualize')
    parser.add_argument("-bounds-file", help="File with bounds data.", type=str,
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\output\\visualize\\ibp.txt')
    parser.add_argument("-img", help="Image to overlay the bounds.", type=str, 
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\mot_benchmark\\train\\ADL-Rundle-6\\img1')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    bounds_file = args.bounds_file
    img = args.img

    bounds_data = np.loadtxt(bounds_file, delimiter=',')
    frame = bounds_data[0][0]

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    fn = os.path.join(img, '%06d.jpg'%(frame))
    im =io.imread(fn)
    height, width = im.shape[:2]
    ax1.imshow(im)
    
    for bounds in bounds_data:
        bounds = np.rint(bounds).astype(np.int64)
        if (bounds[6] < 0 or bounds[6] > width) and (bounds[7] < 0 or bounds[7] > width):
            continue
        if (bounds[8] < 0 or bounds[8] > height) and (bounds[9] < 0 or bounds[9] > height):
            continue
        ax1.add_patch(patches.Rectangle((bounds[6],bounds[7]),bounds[8]-bounds[6],bounds[9]-bounds[7],fill=False,lw=1,ec='red'))
    
    fig.canvas.flush_events()
    plt.draw()
    plt.show(block=True)