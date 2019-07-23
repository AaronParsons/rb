#! /usr/bin/env python
from __future__ import print_function
import pylab as plt
import tensorflow as tf
import sys, cv2, numpy as np
import cPickle, os
from matplotlib.colors import Normalize
import train, find

LABEL = 'ball'
PKL_FILE = '{LABEL}.pkl'.format(LABEL=LABEL)
VERSION = 'v004'
CNN_FILE = '{LABEL}.{VERSION}.ckpl'.format(LABEL=LABEL, VERSION=VERSION)

def read_frame(cap, frame_num):
    deltas = range(-30,31,10)
    frames = [train.read_frame(cap, frame_num+d) for d in deltas]
    baseline = np.median(frames, axis=0)
    delta = frames[len(deltas)/2].astype(np.float) - baseline.astype(np.float)
    return frames[len(deltas)/2], delta

def get_colors(delta, cmap='cool'):
    delta_abs = np.abs(delta)
    alpha = delta_abs / np.max(delta_abs)
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = Normalize()(np.log10(np.abs(delta)))
    colors = cmap(colors)
    colors[...,-1] = alpha
    return colors

labels = {}
try:
    pickle_file = open(PKL_FILE,'r')
    labels[LABEL] = cPickle.load(pickle_file)
except(IOError):
    labels[LABEL] = []

print(labels)

filename = sys.argv[-1]
basename = os.path.basename(filename)
with tf.Session() as session:
    finder = find.FinderCNN(session, CNN_FILE)
    cap = cv2.VideoCapture(filename)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = np.random.randint(nframes)
    #frame_num = 157697
    frame, delta = read_frame(cap, frame_num)
    colors = get_colors(delta)
    frame_plt = plt.imshow(frame, cmap='gray')
    color_plt = plt.imshow(colors)
    centers = np.array(finder.find(frame))
    find_plt = plt.plot(centers[:,0], centers[:,1], 'm+')[0]
    plt.title(basename + ' %d/%d' % (frame_num, nframes))

    def click(event):
        if event.button == 1:
            y,x = int(np.around(event.xdata)), int(np.around(event.ydata))
            labels[LABEL] = labels.get(LABEL, []) + [(basename, frame_num, x, y)]
        elif event.button == 3:
            labels[LABEL] = labels.get(LABEL, [None])[:-1]
        print(labels[LABEL])

    def press(event):
        global frame_num
        if event.key == 's': # Save label file
            print('Writing labels to', PKL_FILE)
            f = open(PKL_FILE, 'w')
            cPickle.dump(labels[LABEL], f)
            f.close()
        if event.key == 'n':
            frame_num = np.random.randint(nframes)
            #frame_num += 1
            frame, delta = read_frame(cap, frame_num)
            frame_plt.set_data(frame)
            colors = get_colors(delta)
            color_plt.set_data(colors)
            centers = np.array(finder.find(frame))
            find_plt.set_xdata(centers[:,0])
            find_plt.set_ydata(centers[:,1])
            plt.title(basename + ' %d/%d' % (frame_num, nframes))
            plt.draw()
        if event.key == 'x':
            labels[LABEL] = labels.get(LABEL, []) + [(basename, frame_num, -1, -1)]
            print(labels[LABEL])

    plt.connect('button_press_event', click)
    plt.connect('key_press_event', press)
    plt.show()
