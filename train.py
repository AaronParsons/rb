import numpy as np
import tensorflow as tf; tf.logging.set_verbosity(tf.logging.ERROR)
from neural import HALF_SZ, x, y_, accuracy, keep_prob, train_step
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

def get_batch(files, labdict, label, label_cnt, non_cnt, half_sz=HALF_SZ):
    '''Provide a batch of labeled images for training a neural net.
    Arguments:
        files: a list of (large) image files
        labdict: a dictionary of labels with filenames as keys and values
            being a {label: [(x0,y0),(x1,y1)]} set of centroids for tagged 
            locations.  This is usually comes from the training/label.pkl file,
            which is built using the sr_tag_imgs.py script.
        label: the label (string) from labdict that we are sampling
        label_cnt: the number of thumbnails matching the label that are to be provided
        non_cnt: the number of thumbnails *not* matching the label that are to be provided
        half_sz: 1/2 the size of the thumbnail width/height.  Default neural.HALF_SZ.
    Returns:
        batch_x: the thumbnails containing both those matching label and not
        batch_y: a one-hot vector that is 1 at each coordinate where batch_x matches
            the label and 0 where it does not.'''
    filemap = {os.path.basename(f): f for f in files if os.path.basename(f) in labdict}
    locs = [(filename, x, y) for filename in labdict.keys() for x,y,L in labdict[filename] if L == label]
    if len(locs) < label_cnt:
        locs *= int(np.ceil(float(label_cnt) / len(locs)))
    sublocs = random.sample(locs, label_cnt)
    has_label = []
    non_label = []
    ratio = float(non_cnt) / float(label_cnt)
    for filename, cx, cy in sublocs:
        filelocs = [(x, y) for f, x, y in locs if f == filename]
        im = cv2.imread(filemap[filename])
        x = cx + random.randint(-half_sz/2, half_sz/2)
        y = cy + random.randint(-half_sz/2, half_sz/2)
        x, y = max(half_sz, x), max(half_sz, y)
        x, y = min(im.shape[1]-half_sz, x), min(im.shape[0]-half_sz, y)
        clip = im[y-half_sz:y+half_sz,x-half_sz:x+half_sz]
        assert clip.shape == (half_sz*2, half_sz*2, 3)
        has_label.append(random_permutation(clip))
        while float(len(non_label)) / float(len(has_label)) < ratio:
            x = random.randint(half_sz, im.shape[1]-half_sz)
            y = random.randint(half_sz, im.shape[0]-half_sz)
            dist = [(x-x0)**2 + (y-y0)**2 for x0,y0 in filelocs]
            if len(dist) == 0 or min(dist) > (1.5 * half_sz)**2:
                clip = im[y-half_sz:y+half_sz,x-half_sz:x+half_sz]
                assert clip.shape == (half_sz*2, half_sz*2, 3)
                non_label.append(random_permutation(clip))
    assert len(has_label) == label_cnt
    non_label = non_label[:non_cnt]
    assert len(non_label) == non_cnt
    batch_x = np.array(has_label + non_label, dtype=np.float32) / 255
    batch_y = np.zeros((batch_x.shape[0],2), dtype=np.float32)
    batch_y[:label_cnt,0] = 1
    batch_y[label_cnt:,1] = 1
    return batch_x, batch_y
        
def train(savefile, files, labdict, label, ncycles, nsamples, 
          half_sz=HALF_SZ, save_every=10, startfile=None):
    '''Trains the neural net in neural.py to identify a label among a set of labeled files.
    Currently, as training proceeds, it moves from a 50/50 ratio of matching/non-matching images
    to a 15/100 ratio that is more reflective of the prior probability in the images (although
    this can be invalidated by only matching over some of the image).
    Arguments:
        savefile: the filename of where to save the coefficients for the neural net.
        files: a list of (large) image files
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
        sess.run(tf.global_variables_initializer())
        for i in range(ncycles):
            label_ratio = max(.5 - float(i) / ncycles, 0.15) # XXX
            label_cnt = int(np.around(label_ratio * nsamples))
            non_cnt = nsamples - label_cnt
            batch_x, batch_y = get_batch(files, labdict, label, label_cnt, non_cnt, half_sz)
            if i % save_every == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                saver.save(sess, os.path.join(os.getcwd(), savefile))
            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
