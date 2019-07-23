from __future__ import print_function
import numpy as np
import tensorflow as tf; tf.logging.set_verbosity(tf.logging.ERROR)
from neural import HALF_SZ, IMG_SIZE, x, y_, accuracy, keep_prob, train_step
import cv2
import random
import os

def random_permutation(img):
    '''Randomly flip or rotate an image.'''
    if random.randint(0, 2) == 0:
        img = np.rot90(img)
    if random.randint(0, 2) == 0:
        img = np.flipud(img)
    if random.randint(0, 2) == 0:
        img = np.fliplr(img)
    return img

def read_frame(cap, frame_num):
    '''Read a frame from a cv2.VideoCapture object.'''
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

def get_thumbs_and_bgs(label_list, movie_file, width=IMG_SIZE, height=IMG_SIZE):
    labels = [(frame_num, x, y) for (basename, frame_num, x, y) in label_list if basename == os.path.basename(movie_file)]
    assert(len(labels) > 0)
    cap = cv2.VideoCapture(movie_file)
    thumbs, bgs = [], []
    # Walk though labels dictionary and create positive and negative files
    for cnt,(frame_num, x, y) in enumerate(labels):
        frame = read_frame(cap, frame_num)
        if x < 0 or y < 0: # negative image with no matches
            bgs.append(frame)
        else:
            x0, y0 = x-width/2, y-height/2
            thumbs.append(frame[x0:x0+width,y0:y0+height].copy())
            # XXX assumes only one match per frame
            frame[x0:x0+width,y0:y0+height] = 128 # blank out match
            bgs.append(frame)
    #return np.array(thumbs), np.array(bgs)
    return np.array(thumbs, dtype=np.float32), np.array(bgs, dtype=np.float32)

def create_nulls(bgs, nsamples, width, height):
    bgs.shape = (-1,) + bgs.shape[-2:]
    ws = np.random.randint(bgs.shape[0], size=nsamples)
    xs = np.random.randint(bgs.shape[1]-width, size=nsamples)
    ys = np.random.randint(bgs.shape[2]-height, size=nsamples)
    vec = np.array([bgs[w,x:x+width,y:y+height] for w,x,y in zip(ws,xs,ys)])
    return vec

def create_samples(thumb, bgs, nsamples, bg_level=255, bg_thresh=150, noise=1, dither=3):
    width,height = thumb.shape[-2:]
    vec = create_nulls(bgs, nsamples, width, height)
    tx, ty = np.where(np.abs(thumb.astype(np.int) - bg_level) > bg_thresh) # where to use thumb
    for i in range(vec.shape[0]):
        dx, dy = tx.copy(), ty.copy()
        #dx, dy = tx - width / 2., ty - height / 2.
        #theta = np.random.uniform(0,2*np.pi)
        #dx,dy = dx*np.cos(theta) - dy*np.sin(theta), dy*np.cos(theta) + dx*np.sin(theta)
        #dx,dy = dx + width / 2., dy + height / 2
        if dither > 0:
            dx = dx + np.random.randint(-dither,dither)
            dy = dy + np.random.randint(-dither,dither)
        if np.random.randint(2): # 90-deg rotation
            dx, dy = dy, dx
        if np.random.randint(2): # flip lr
            dx = width - 1 - dx
        if np.random.randint(2): # flip ud
            dy = height - 1 - dy
        #dx = np.around(dx).clip(0,width-1).astype(np.int)
        #dy = np.around(dy).clip(0,height-1).astype(np.int)
        dx = dx.clip(0,width-1).astype(np.int)
        dy = dy.clip(0,height-1).astype(np.int)
        vec[i,dx,dy] = thumb[tx,ty]
    if noise > 0:
        #vec += np.random.randint(noise, size=vec.shape, dtype=vec.dtype)
        vec += np.random.normal(noise, size=vec.shape)
    return vec

def get_batch(files, label_list, label_cnt, non_cnt, half_sz=HALF_SZ):
    '''Provide a batch of labeled images for training a neural net.
    Arguments:
        files: list movie files
        labels: a dictionary of labels with filenames as keys and values
            being a {label: [(x0,y0),(x1,y1)]} set of centroids for tagged 
            locations.
        label: the label (string) from labdict that we are sampling
        label_cnt: the number of thumbnails matching the label that are to be provided
        non_cnt: the number of thumbnails *not* matching the label that are to be provided
        half_sz: 1/2 the size of the thumbnail width/height.  Default neural.HALF_SZ
    Returns:
        batch_x: the thumbnails containing both those matching label and not
        batch_y: a one-hot vector that is 1 at each coordinate where batch_x matches
            the label and 0 where it does not.'''
    thumbs, bgs = [], []
    if type(files) == str:
        files = [files]
    for f in files:
        _thumbs, _bgs = get_thumbs_and_bgs(label_list, f, width=IMG_SIZE, height=IMG_SIZE)
        thumbs.append(_thumbs)
        bgs.append(_bgs)
    thumbs = np.concatenate(thumbs, axis=0)
    bgs = np.concatenate(bgs, axis=0)
    num_per = int(np.ceil(float(label_cnt) / len(thumbs)))
    has_label = [create_samples(thumb, bgs, num_per, bg_level=255, bg_thresh=150, noise=1, dither=3) for thumb in thumbs]
    has_label = np.concatenate(has_label, axis=0)[:label_cnt]
    non_label = create_nulls(bgs, non_cnt, width=IMG_SIZE, height=IMG_SIZE)
    #np.savez('out.npz', ball=has_label, no_ball=non_label)
    ratio = float(non_cnt) / float(label_cnt)
        #has_label.append(random_permutation(clip))
        #non_label.append(random_permutation(clip))
    batch_x = np.concatenate([has_label, non_label], axis=0).astype(np.float32) / 255
    #batch_x = np.array(has_label + non_label, dtype=np.float32) / 255
    batch_x.shape = (-1, IMG_SIZE, IMG_SIZE, 1)
    batch_y = np.zeros((batch_x.shape[0],2), dtype=np.float32)
    batch_y[:label_cnt,0] = 1
    batch_y[label_cnt:,1] = 1
    return batch_x, batch_y
        
def train(savefile, files, label_list, ncycles, nsamples, 
          half_sz=HALF_SZ, save_every=10, startfile=None):
    '''Trains the neural net in neural.py to identify a label among a set of labeled files.
    Currently, as training proceeds, it moves from a 50/50 ratio of matching/non-matching images
    to a 15/100 ratio that is more reflective of the prior probability in the images (although
    this can be invalidated by only matching over some of the image).
    Arguments:
        savefile: the filename of where to save the coefficients for the neural net.
        files: a list of movie files
        labdict: a dictionary of labels with filenames as keys and values
            being a {label: [(x0,y0),(x1,y1)]} set of centroids for tagged 
            locations.  This is usually comes from the training/label.pkl file,
            which is built using the sr_tag_imgs.py script.
        label: the label (string) from labdict that we are sampling
        ncycles: the number of training cycles to run
        nsamples: the number of thumbnails to train in per cycle
        half_sz: 1/2 the size of the thumbnail width/height.  Default: neural.HALF_SZ
        save_every: the number of cycles between saves.  Default 10.
        startfile: the (optional) save file from a previous run of train, so that
            fitting picks up where it left off.
    Returns:
        None
    '''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if startfile is not None:
            print('Restoring from', startfile)
            saver.restore(sess, startfile)
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(ncycles):
            #label_ratio = max(.5 - float(i) / ncycles, 0.15) # XXX
            label_ratio = .15
            label_cnt = int(np.around(label_ratio * nsamples))
            non_cnt = nsamples - label_cnt
            batch_x, batch_y = get_batch(files, label_list, label_cnt, non_cnt, half_sz)
            if i % save_every == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                saver.save(sess, os.path.join(os.getcwd(), savefile))
            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
