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
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\output\\visualize\\crown-every-frame-perturbed.txt')
    parser.add_argument("-img", help="Image to overlay the bounds.", type=str, 
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\mot_benchmark\\train\\ADL-Rundle-6\\img1')
    return parser.parse_args()

def convert_x_to_bbox(bounds,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    x_l = bounds[2]
    y_l = bounds[3]
    x_u = bounds[5]
    y_u = bounds[6]
    s = bounds[7]
    r = bounds[8]

    w = np.sqrt(s * r)
    h = s / w

    bbox = np.array([x_l-w/2,y_l-h/2,x_u+w/2,y_u+h/2])
    return bbox

index = 0

if __name__ == '__main__':
    args = parse_args()
    bounds_file = args.bounds_file
    img = args.img

    bounds_data = np.loadtxt(bounds_file, delimiter=',', dtype=np.float32)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    
    def display_frame(i):
        ax1.clear()
        bounds = bounds_data[i]
        frame = bounds[0]
        fn = os.path.join(img, '%06d.jpg'%(frame))
        im =io.imread(fn)
        height, width = im.shape[:2]
        ax1.imshow(im)
        plt.title('Frame ' + str(frame))

        bbox = convert_x_to_bbox(bounds,score=None)

        ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,lw=1,ec='red'))

        #draw lines connecting their corners
        '''
        u_corners = [
            (bbox_u[0], bbox_u[1]),
            (bbox_u[0], bbox_u[3]),
            (bbox_u[2], bbox_u[1]),
            (bbox_u[2], bbox_u[3])
        ]

        l_corners = [
            (bbox_l[0], bbox_l[1]),
            (bbox_l[0], bbox_l[3]),
            (bbox_l[2], bbox_l[1]),
            (bbox_l[2], bbox_l[3])
        ]
        for pt1, pt2 in zip(l_corners, u_corners):
            ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', lw=1)
        '''
        
        #draw the actual detection
        ax1.add_patch(patches.Rectangle((bounds[9],bounds[10]),bounds[11]-bounds[9],bounds[12]-bounds[10],fill=False,lw=1,ec='green'))

        fig.canvas.flush_events()
        plt.draw()

    def change_frame(event):
        global index
        if event.key == 'right':
            index = (index + 1) % len(bounds_data)  # Move to the next frame
        elif event.key == 'left':
            index = (index - 1) % len(bounds_data)  # Move to the previous frame
        display_frame(index)

    display_frame(index)
    fig.canvas.mpl_connect('key_press_event', change_frame)
    plt.show()