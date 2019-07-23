#! /usr/bin/env python
from __future__ import print_function
import sys, cv2, numpy as np
import cPickle, os, glob, vec

def read_frame(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    frame_curr = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    return frame_curr

#def create_samples(thumb, bgs, nsamples, bg_level=255, bg_thresh=150, noise=10, dither=3):
def create_samples(thumb, bgs, nsamples, bg_level=255, bg_thresh=150, noise=0, dither=0):
    width,height = thumb.shape[-2:]
    bgs.shape = (-1,) + bgs.shape[-2:]
    tx, ty = np.where(np.abs(thumb.astype(np.int) - bg_level) > bg_thresh) # where to use thumb
    ws = np.random.randint(bgs.shape[0], size=nsamples)
    xs = np.random.randint(bgs.shape[1]-width, size=nsamples)
    ys = np.random.randint(bgs.shape[2]-height, size=nsamples)
    vec = np.array([bgs[w,x:x+width,y:y+height] for w,x,y in zip(ws,xs,ys)])
    for i in range(vec.shape[0]):
        dx, dy = tx.copy(), ty.copy()
        #dx, dy = tx - width / 2., ty - height / 2.
        #theta = np.random.uniform(0,2*np.pi)
        #dx,dy = dx*np.cos(theta) - dy*np.sin(theta), dy*np.cos(theta) + dx*np.sin(theta)
        #dx,dy = dx + width / 2., dy + height / 2
        #dx = dx + np.random.randint(-dither,dither)
        #dy = dy + np.random.randint(-dither,dither)
        #if np.random.randint(2): # 90-deg rotation
        #    dx, dy = dy, dx
        #if np.random.randint(2): # flip lr
        #    dx = width - 1 - dx
        #if np.random.randint(2): # flip ud
        #    dy = height - 1 - dy
        #dx = np.around(dx).clip(0,width-1).astype(np.int)
        #dy = np.around(dy).clip(0,height-1).astype(np.int)
        dx = dx.clip(0,width-1).astype(np.int)
        dy = dy.clip(0,height-1).astype(np.int)
        vec[i,dx,dy] = thumb[tx,ty]
    #vec += np.random.randint(noise, size=vec.shape, dtype=vec.dtype)
    return vec

WIDTH = 21
HEIGHT = 21
NUM_VECS = 900
LABEL = 'ball'
PKL_FILE = LABEL + '.pkl'

filelist = sys.argv[1:]
labels = {}
try:
    pickle_file = open(PKL_FILE,'r')
    labels[LABEL] = cPickle.load(pickle_file)
except(IOError):
    labels[LABEL] = []

#print(labels)

filename = sys.argv[-1]
cap = cv2.VideoCapture(filename)

if not os.path.exists(LABEL): os.mkdir(LABEL)

vecfile = '%s.vec' % (LABEL)
if not os.path.exists(LABEL+'/neg'):
    os.mkdir(LABEL+'/neg')

thumbs = []
frames = []
# Walk though labels dictionary and create positive and negative files
for cnt,(basename, frame_num, x, y) in enumerate(labels[LABEL]):
    if basename != os.path.basename(filename):
        continue
    frame = read_frame(cap, frame_num)
    print(frame.shape, x, y)
    if x < 0 or y < 0: # negative image with no matches
        cv2.imwrite(LABEL+'/neg/%06d.jpg' % (frame_num), frame)
    else:
        x0, y0 = x-WIDTH/2, y-HEIGHT/2
        thumb = frame[x0:x0+WIDTH,y0:y0+HEIGHT].copy()
        #cv2.imwrite('pos/%06d.jpg' % (cnt), thumb)
        thumbs.append(thumb)
        frame[x0:x0+WIDTH,y0:y0+HEIGHT] = 128 # blank out ball
        frames.append(frame)
        cv2.imwrite(LABEL+'/neg/%06d.jpg' % (frame_num), frame) # re-use as negative

# Write background file for use with opencv_createsamples
negatives = glob.glob(LABEL+'/neg/*.jpg')
negatives += glob.glob('negs/*.jpg')[:1000]
f = open('bg.txt','w')
f.write('\n'.join(negatives))
f.close()
NUM_PER_POS = NUM_VECS / len(thumbs)
frames = np.array(frames)
samples = np.concatenate([create_samples(thumb, frames, NUM_PER_POS) for thumb in thumbs], axis=0)
vec.save_vec(samples, vecfile)

#if not os.path.exists(vecfile):
#    # Create positive images on backgrounds
#    for pos in positives:
#        params = {'filebase':pos[:-4], 'maxxangle': 1.1, 'maxyangle': 1.1, 'maxzangle': 3, 'num_per_pos': NUM_PER_POS,
#                  'bgcolor': 255, 'bgthresh': 127, 'WIDTH':WIDTH, 'HEIGHT':HEIGHT}
#        # Create a bunch of *.vec files
#        cmd = 'opencv_createsamples -img {filebase}.jpg -bg bg.txt -info {filebase}.ann -pngoutput -maxxangle {maxxangle} -maxyangle {maxyangle} maxzangle {maxzangle} -num {num_per_pos} -bgcolor {bgcolor} -bgthresh {bgthresh} -w {WIDTH} -h {HEIGHT} -vec {filebase}.vec'.format(**params)
#        print(cmd)
#        os.system(cmd)
#    vecfiles = glob.glob('pos/*.vec')
#    merge_vecs(vecfiles, vecfile)

OUTDIR = LABEL+'/lbp_class_%s' % (LABEL)
if not os.path.exists(OUTDIR):
    # Train the classifier
    num_neg = len(open('bg.txt').readlines())
    #positives = glob.glob('pos/*.jpg')
    #num_pos = int(len(positives) * NUM_PER_POS * .85)
    num_pos = int(NUM_VECS * .85)
    os.mkdir(OUTDIR)
    train_params = {'NUMPOS':num_pos, 'NUMNEG':num_neg, 'WIDTH':WIDTH, 'HEIGHT':HEIGHT, 'LABEL':LABEL}
    cmd = 'opencv_traincascade -data {LABEL}/lbp_class_{LABEL} -vec {LABEL}.vec -bg bg.txt -precalcValBufSize 0 -precalcIdxBufSize 0 -numPos {NUMPOS} -numNeg {NUMNEG} -numStages 14 -minHitRate 0.999 -maxfalsealarm 0.5 -w {WIDTH} -h {HEIGHT} -nonsym -baseFormatSave -featureType LBP -acceptanceRatioBreakValue 1e-5'.format(**train_params)
    print(cmd)
    #import IPython; IPython.embed()
    os.system(cmd)
#os.remove('bg.txt')
#os.remove(vecfile)
