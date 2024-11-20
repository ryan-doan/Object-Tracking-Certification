import os
import cv2
from PIL import Image

fps = 30
path = 'visualize_output'
curr_frame_index = 1
frame_img_name = '%06d.png'%curr_frame_index

first_image = cv2.imread(os.path.join(path, frame_img_name))
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
video = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

while os.path.isfile(os.path.join(path, frame_img_name)):
    frame = cv2.imread(os.path.join(path, frame_img_name))
    video.write(frame)
    curr_frame_index += 1
    frame_img_name = '%06d.png'%curr_frame_index

video.release()