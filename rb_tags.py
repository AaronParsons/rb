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
    deltas = [-20,-10,0,10,20]
    frames = []
    for delta in deltas:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1-delta)
        frames.append(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY))
    baseline = np.median(frames, axis=0)
    delta = frames[2].astype(np.float) - baseline.astype(np.float)
    return frames[2], delta

def get_colors(delta, cmap='jet'):
    delta_abs = np.abs(delta)
    alpha = delta_abs / np.max(delta_abs)
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = Normalize()(np.log10(np.abs(delta)))
    colors = cmap(colors)
    colors[...,-1] = alpha
    return colors

def find_balls(ballfinder, frame):
    if ballfinder is None: return []
    return ballfinder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(WIDTH,HEIGHT), maxSize=(WIDTH*2,HEIGHT*2), flags=cv2.CASCADE_SCALE_IMAGE)
    #return ballfinder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(WIDTH,HEIGHT), flags=cv2.CASCADE_SCALE_IMAGE)
    #return ballfinder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, flags=cv2.CASCADE_SCALE_IMAGE)
    #for (x,y,w,h) in balls:
    #    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    #    ax = plt.gca()
    #    ax.add_patch(rect)

WIDTH = 21
HEIGHT = 21
LABEL = 'ball'
PKL_FILE = LABEL + '.pkl'
BALL_CASCADE = '{LABEL}/lbp_class_{LABEL}/cascade.xml'.format(LABEL='ball')
print('Reading', BALL_CASCADE)
if os.path.exists(BALL_CASCADE):
    ballfinder = cv2.CascadeClassifier(BALL_CASCADE)
else:
    ballfinder = None

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
frame_num = 157697
print(basename, nframes)
frame, delta = read_frame(cap, frame_num)
colors = get_colors(delta)
#frame_plt = plt.imshow(frame, cmap='gray')
color_plt = plt.imshow(colors)
balls = find_balls(ballfinder, frame)
ball_plt1 = plt.plot([b[0] for b in balls], [b[1] for b in balls], 'y+')[0]
ball_plt2 = plt.plot([b[0]+b[2] for b in balls], [b[1]+b[3] for b in balls], 'g+')[0]
plt.title(frame_num)
#plt.show()

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
        #frame_num = np.random.randint(nframes)
        frame_num += 1
        frame, delta = read_frame(cap, frame_num)
        colors = get_colors(delta)
        #frame_plt.set_data(frame)
        color_plt.set_data(colors)
        balls = find_balls(ballfinder, frame)
        ball_plt1.set_xdata([b[0] for b in balls])
        ball_plt1.set_ydata([b[1] for b in balls])
        ball_plt2.set_xdata([b[0]+b[2] for b in balls])
        ball_plt2.set_ydata([b[1]+b[3] for b in balls])
        plt.title(frame_num)
        plt.draw()
    if event.key == 'x':
        labels[LABEL] = labels.get(LABEL, []) + [(basename, frame_num, -1, -1)]
        print(labels[LABEL])

plt.connect('button_press_event', click)
plt.connect('key_press_event', press)
plt.show()
