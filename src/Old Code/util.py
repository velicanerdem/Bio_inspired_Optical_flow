import cv2
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import join
import pandas as pd
import time

"""
Can use notebook instead: %time %timeit
"""
# class Timer:
    # def __init__(self, msg='Time elapsed'):
        # self.msg = msg
    # def __enter__(self):
        # self.start = time.time()
        # return self
    # def __exit__(self, *args):
        # self.end = time.time()
        # duration = self.end - self.start
        # print(f'{self.msg}: {duration:.2f}s')

"""
Below are not used
"""

# class Event:
    # __slots__ = 't', 'x', 'y', 'p'
    # def __init__(self, t, x, y, p):
        # self.t = t
        # self.x = x
        # self.y = y
        # self.p = p
    # def __repr__(self):
        # return f'Event(t={self.t:.3f}, x={self.x}, y={self.y}, p={self.p})'    
    
    
# def normalize_image(image, percentile_lower=1, percentile_upper=99):
    # mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    # if mini == maxi:
        # return 0 * image + 0.5  # gray image
    # return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


# class EventData:
    # def __init__(self, event_list, width, height):
        # self.event_list = event_list
        # self.width = width
        # self.height = height

    # def add_frame_data(self, data_folder, max_frames=100):
        # timestamps = np.genfromtxt(join(data_folder, 'image_timestamps.txt'), max_rows=int(max_frames))
        # frames = []
        # frame_timestamps = []
        # with open(join(data_folder, 'image_timestamps.txt')) as f:
            # for line in f:
                # fname, timestamp = line.split(' ')
                # timestamp = float(timestamp)
                # frame = cv2.imread(join(data_folder, fname), cv2.IMREAD_GRAYSCALE)
                # if not (frame.shape[0] == self.height and frame.shape[1] == self.width):
                    # continue
                # frames.append(frame)
                # frame_timestamps.append(timestamp)
                # if timestamp >= self.event_list[-1].t:
                    # break
        # frame_stack = normalize_image(np.stack(frames, axis=0))
        # self.frames = [f for f in frame_stack]
        # self.frame_timestamps = frame_timestamps
    
"""
Didn't try using it.
"""
# def animate(images, fig_title=''):
    # fig = plt.figure(figsize=(0.1, 0.1))  # don't take up room initially
    # fig.suptitle(fig_title)
    # fig.set_size_inches(7.2, 5.4, forward=False)  # resize but don't update gui
    # ims = []
    # for image in images:
        # im = plt.imshow(normalize_image(image), cmap='gray', vmin=0, vmax=1, animated=True)
        # ims.append([im])
    # ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    # plt.close(ani._fig)
    # return ani
