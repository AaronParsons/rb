import train
import sys, cPickle

LABEL = 'ball'
LABELS = LABEL+'.pkl'
OUTFILE = LABEL+'.v001.ckpl'
MOVIE_FILE = '/Users/aparsons/Movies/rb/rb071919_parsons_johnson.mp4'
STARTFILE = None

label_list = cPickle.load(open(LABELS))
train.train(OUTFILE, MOVIE_FILE, label_list, 20000, 100, startfile=STARTFILE)
