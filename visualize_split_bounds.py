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
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\output\\visualize\\crown-split-ptb-set.txt')
    parser.add_argument("-img", help="Image to overlay the bounds.", type=str, 
                        default='C:\\Users\\doann\\Documents\\lirpa\\PyTorch_Kalman\\mot_benchmark\\train\\ADL-Rundle-6\\img1')
    return parser.parse_args()

def convert_x_to_bbox(bounds,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    x_l = bounds[0]
    y_l = bounds[1]
    s_l = bounds[2]
    x_u = bounds[3]
    y_u = bounds[4]
    s_u = bounds[5]
    r = bounds[6]

    w_l = np.sqrt(s_l * r)
    h_l = s_l / w_l
    w_u = np.sqrt(s_u * r)
    h_u = s_u / w_u

    bbox_l = np.array([x_l-w_l/2,y_l-h_l/2,x_u+w_l/2,y_u+h_l/2])
    bbox_u = np.array([x_l-w_u/2,y_l-h_u/2,x_u+w_u/2,y_u+h_u/2])
    return bbox_l, bbox_u

index = 0

if __name__ == '__main__':
    args = parse_args()
    bounds_file = args.bounds_file
    img = args.img

    bounds_data = np.loadtxt(bounds_file, delimiter=',', dtype=np.float32)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

    def draw_bounds(ax1, bbox):
        ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,lw=1,ec='red'))

    def display_frame(i):
        ax1.clear()
        bounds = bounds_data[i]
        frame = bounds[0]
        fn = os.path.join(img, '%06d.jpg'%(frame))
        im =io.imread(fn)
        height, width = im.shape[:2]
        ax1.imshow(im)
        plt.title('Frame ' + str(frame))

        bbox_l, bbox_u = convert_x_to_bbox(bounds[2:9],score=None)

        draw_bounds(ax1, bbox_l)
        draw_bounds(ax1, bbox_u)

        #bbox_l, bbox_u = convert_x_to_bbox(bounds[9:16],score=None)

        #draw_bounds(ax1, bbox_l)
        #draw_bounds(ax1, bbox_u)

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
        ax1.add_patch(patches.Rectangle((bounds[16],bounds[17]),bounds[18]-bounds[16],bounds[19]-bounds[17],fill=False,lw=1,ec='green'))

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