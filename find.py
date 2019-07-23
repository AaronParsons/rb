from __future__ import print_function
import cv2
from . import neural
import tensorflow as tf; tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

#W,H = 48,36

#MTG_BACK_CLASSIFIER = cv2.CascadeClassifier(CASCADE_MTG_BACK)
#CASCADE_MTG_BACK = 'data/cascades/cascade_mtg_back003.xml'
#
#def find_from_file(filename, classifier=MTG_BACK_CLASSIFIER, min_size=(W,H), scale_factor=1.1, min_neighbors=1):
#    im = cv2.imread(filename)
#    return find(im, classifier=classifier, min_size=min_size, scale_factor=scale_factor, min_neighbors=min_neighbors)
#
#def find(im, classifier=MTG_BACK_CLASSIFIER, min_size=(W,H), scale_factor=1.1, min_neighbors=1):
#    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#    cards = classifier.detectMultiScale(gray, scaleFactor=scale_factor, 
#                minNeighbors=min_neighbors, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)
#    centers = [(x+w/2,y+h/2) for (x,y,w,h) in cards]
#    return centers

class FinderCNN:
    '''A class for using a convolutional neural net to find matches within an image.'''
    def __init__(self, session, savefile, verbosity=1):
        '''Arguments:
            session: a tensorflow session
            savefile: the filename where the neural net coefficients are stored.
            verbosity: the level of verbosity in printing things to screen.'''
        self.verbosity = verbosity
        saver = tf.train.Saver()
        if self.verbosity >= 1: print('Restoring from', savefile)
        saver.restore(session, savefile)
        self.session = session
        self.half_sz = neural.HALF_SZ
        self.x = neural.x
        self.y = neural.y_conv
        self.keep_prob = neural.keep_prob
    def _run(self, x_in):
        k = self.session.run([tf.nn.softmax(self.y)], feed_dict={self.x:x_in, self.keep_prob:1.})
        has_label = np.argmax(np.array(k)[0], 1) == 0
        return has_label
    def gen_centers(self, shape, step, xrng=None, yrng=None):
        '''Create a set of center locations to sampling an image to check for matches.
        Arguments:
            shape: the shape of the image
            step: the separation in pixels between centers in both the x and y directions
            xrng: the range of the image to sample in the x direction
            yrng: the range of the image to sample in the y direction
        Returns:
            centers: a list of (x,y) locations of centers'''
        assert len(shape) == 2
        if xrng is None: xrng = (0, shape[1])
        if yrng is None: yrng = (0, shape[0])
        xmin = max(xrng[0], self.half_sz)
        xmax = min(xrng[1], shape[1] - self.half_sz + 1)
        ymin = max(yrng[0], self.half_sz)
        ymax = min(yrng[1], shape[0] - self.half_sz + 1)
        centers = [(cx,cy) for cy in range(ymin, ymax, step) for cx in range(xmin, xmax, step)]
        return centers
    def get_xin(self, im, centers):
        '''Generate a cube of thumbnails to feed the neural net.
        Arguments:
            im: the image to sample
            centers: the center points at which to sample
        Returns:
            x_in: a cube of thumbnails.'''
        x_in = np.array([im[cy-self.half_sz:cy+self.half_sz,cx-self.half_sz:cx+self.half_sz] for cx,cy in centers])
        return x_in.astype(np.float) / 255
    def find(self, im, xrng=(0,120), yrng=(0,200), step=24):
        '''Sample an image and return locations that are matches.
        Arguments:
            im: the image to sample
            xrng: the range of the image to sample in the x direction.  Default (0,120)
            yrng: the range of the image to sample in the y direction.  Default (0,200)
            step: the separation in pixels between centers in both the x and y directions.  Default 24.
        Returns:
            centers: centers of locations where a match occured.'''
        centers = self.gen_centers(im.shape[:-1], step, xrng=xrng, yrng=yrng)
        x_in = self.get_xin(im, centers)
        has_label = self._run(x_in)
        return [c for c,h in zip(centers, has_label) if h]
        
