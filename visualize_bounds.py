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

index = 0

if __name__ == '__main__':
    args = parse_args()
    bounds_file = args.bounds_file
    img = args.img

    bounds_data = np.loadtxt(bounds_file, delimiter=',')

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
        bounds = np.rint(bounds).astype(np.int64)

        #draw lower and upper bbox
        if (bounds[6] < 0 or bounds[6] > width) and (bounds[7] < 0 or bounds[7] > width):
            pass
        elif (bounds[8] < 0 or bounds[8] > height) and (bounds[9] < 0 or bounds[9] > height):
            pass
        else:
            ax1.add_patch(patches.Rectangle((bounds[6],bounds[7]),bounds[8]-bounds[6],bounds[9]-bounds[7],fill=False,lw=1,ec='red'))

        if (bounds[2] < 0 or bounds[2] > width) and (bounds[3] < 0 or bounds[3] > width):
            pass
        elif (bounds[4] < 0 or bounds[4] > height) and (bounds[5] < 0 or bounds[5] > height):
            pass
        else:
            ax1.add_patch(patches.Rectangle((bounds[2],bounds[3]),bounds[4]-bounds[2],bounds[5]-bounds[3],fill=False,lw=1,ec='red'))

        #draw lines connecting their corners
        u_corners = [
            (bounds[6], bounds[7]),
            (bounds[6], bounds[9]),
            (bounds[8], bounds[7]),
            (bounds[8], bounds[9])
        ]

        l_corners = [
            (bounds[2], bounds[3]),
            (bounds[2], bounds[5]),
            (bounds[4], bounds[3]),
            (bounds[4], bounds[5])
        ]

        for pt1, pt2 in zip(l_corners, u_corners):
            ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', lw=1)

        #draw the actual detection
        ax1.add_patch(patches.Rectangle((bounds[10],bounds[11]),bounds[12]-bounds[10],bounds[13]-bounds[11],fill=False,lw=1,ec='green'))

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