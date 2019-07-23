#! /usr/bin/env python
from __future__ import print_function
import pylab as plt
import sys, cv2, numpy as np
import cPickle, os
from matplotlib.colors import Normalize
import matplotlib.patches as patches

def read_frame(cap, frame_num):
    #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-2)
    #$frame_prev = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float)
    #frame_curr = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    #frame_next = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float)
    #delta = frame_curr.astype(np.float) - 0.5 * (frame_prev + frame_next)
    #return frame_curr, delta
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-3)
    frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY) for i in range(5)]
    baseline = np.median(frames, axis=0)
    delta = frames[2].astype(np.float) - baseline.astype(np.float)
    return frames[2], delta

def get_colors(delta, cmap='jet'):
    delta_abs = np.abs(delta)
    alpha = delta_abs / np.max(delta_abs)
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = Normalize()(-delta)
    colors = cmap(colors)
    colors[...,-1] = alpha
    return colors

WIDTH = 21
HEIGHT = 21
LABEL = 'ball'
PKL_FILE = LABEL + '.pkl'
BALL_CASCADE = '{LABEL}/lbp_class_{LABEL}/cascade.xml'.format(LABEL='ball')
print('Reading', BALL_CASCADE)
ballfinder = cv2.CascadeClassifier(BALL_CASCADE)

filelist = sys.argv[1:]
labels = {}
try:
    pickle_file = open(PKL_FILE,'r')
    labels[LABEL] = cPickle.load(pickle_file)
except(IOError):
    labels[LABEL] = []

print(labels)

filename = sys.argv[-1]
basename = os.path.basename(filename)
cap = cv2.VideoCapture(filename)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = np.random.randint(nframes)
#frame_num = 157697
print(basename, nframes)
frame, delta = read_frame(cap, frame_num)
#balls = ballfinder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(WIDTH,HEIGHT), maxSize=(WIDTH,HEIGHT), flags=cv2.CASCADE_SCALE_IMAGE)
balls = ballfinder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(WIDTH/2,HEIGHT/2), maxSize=(WIDTH*2,HEIGHT*2), flags=cv2.CASCADE_SCALE_IMAGE)
colors = get_colors(delta)
frame_plt = plt.imshow(frame, cmap='gray')
color_plt = plt.imshow(colors)
for (x,y,w,h) in balls:
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
plt.title(frame_num)
#plt.show()

#lab_plots = {}


#def update_plot(plts, label):
#    global lab_plots
#    data = np.array([(x,y) for x,y,L in labdict[curfile] if L == label])
#    if data.size == 0:
#        data = np.empty((0,2))
#    print(label, data)
#    if plts.has_key(label):
#        lplt = lab_plots[label]
#        lplt.set_xdata(data[:,0])
#        lplt.set_ydata(data[:,1])
#    else:
#        labels[label] = len(labels)
#        color = colors[labels[label] % len(colors)]
#        lab_plots[label] = plt.plot(data[:,0], data[:,1], color)[0]
#    return

#for label in labels.keys():
#    update_plot(lab_plots, label)
#
#cur_label = get_cur_label()

def click(event):
    if event.button == 1:
        y,x = int(np.around(event.xdata)), int(np.around(event.ydata))
        labels[LABEL] = labels.get(LABEL, []) + [(basename, frame_num, x, y)]
    elif event.button == 3:
        labels[LABEL] = labels.get(LABEL, [None])[:-1]
    print(labels[LABEL])

def press(event):
    global frame_num
    if event.key == 'n':
        print('Writing labels')
        f = open(PKL_FILE, 'w')
        cPickle.dump(labels[LABEL], f)
        f.close()
        frame_num = np.random.randint(nframes)
        frame, delta = read_frame(cap, frame_num)
        colors = get_colors(delta)
        frame_plt.set_data(frame)
        color_plt.set_data(colors)
        plt.title(frame_num)
        plt.draw()
    if event.key == 'x':
        labels[LABEL] = labels.get(LABEL, []) + [(basename, frame_num, -1, -1)]
        print(labels[LABEL])
        #curfile = os.path.basename(filelist[filecnt])
        #for label in labels.keys():
        #    update_plot(lab_plots, label)
        #plt.draw()
    #if event.key == 'y':
    #    cur_label = get_cur_label()
    #    if cur_label not in labels:
    #        labels[cur_label] = len(labels)

plt.connect('button_press_event', click)
plt.connect('key_press_event', press)
plt.show()
