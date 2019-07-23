#! /usr/bin/env python
from __future__ import print_function
import pylab as plt
import os, sys, cv2, numpy as np
import tensorflow as tf
import train, find

LABEL = 'ball'
VERSION = 'v001'
CNN_FILE = '{LABEL}.{VERSION}.ckpl'.format(LABEL=LABEL, VERSION=VERSION)

filename = sys.argv[-1]
basename = os.path.basename(filename)
with tf.Session() as session:
    finder = find.FinderCNN(session, CNN_FILE)
    cap = cv2.VideoCapture(filename)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = np.random.randint(nframes)
    frame_num = 157697
    frame = train.read_frame(cap, frame_num)
    frame_plt = plt.imshow(frame, cmap='gray')
    centers = np.array(finder.find(frame))
    find_plt = plt.plot(centers[:,0], centers[:,1], 'm+')[0]
    plt.title(basename + ' %d/%d' % (frame_num, nframes))

    def press(event):
        global frame_num
        if event.key == 'n':
            #frame_num = np.random.randint(nframes)
            frame_num += 1
            frame = train.read_frame(cap, frame_num)
            frame_plt.set_data(frame)
            centers = np.array(finder.find(frame))
            find_plt.set_xdata(centers[:,0])
            find_plt.set_ydata(centers[:,1])
            plt.title(basename + ' %d/%d' % (frame_num, nframes))
            plt.draw()

    plt.connect('key_press_event', press)
    plt.show()
