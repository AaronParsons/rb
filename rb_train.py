import train
import sys, cPickle

LABEL = 'ball'
LABELS = LABEL+'.pkl'
OUTFILE = LABEL+'.v005.ckpl'
MOVIE_FILE = '/Users/aparsons/Movies/rb/rb071919_parsons_johnson.mp4'
STARTFILE = LABEL+'.v004.ckpl'

label_list = cPickle.load(open(LABELS))
train.train(OUTFILE, MOVIE_FILE, label_list, 2000, 400, startfile=STARTFILE)
