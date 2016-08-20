# import sys
# import os.path
# sys.path.append(
#             os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from neuralforest import ShallowNeuralForest

import csv
import os
import random 
from scipy import misc
import numpy as np
from six.moves import cPickle as pkl 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

FD = os.path.dirname(os.path.realpath(__file__))
patch_dim = 70

def _analyze_bbox(dat):
    """Given a list of bounding boxes, return the min/max sizes of the boxes.
    Assumes .dat format: <name> \t <num_brick> \t <x,y,w,h>"""
    with open(dat, 'rb') as f:
        reader = csv.reader(f,delimiter='\t')
        minW, minH, maxW, maxH = 10e8, 10e8, -1, -1 
        for row in reader:
            bbox =  row[2].split(' ')
            _,_,w,h = map(int, bbox)
            minW, minH, maxW, maxH = min(minW,w), min(minH,h), max(maxW,w), max(maxH,h)
        return minW, minH, maxW, maxH

def _jiggle_bbox(img, bbox, size, jig_amt = 5):
    """Given an image and the bounding box of brick, return a image slice of size <size>
    that contains the brick but allows random jiggling.
    Default jiggle amount is less than 5 pixels"""
    n = random.randint(-jig_amt, jig_amt)
    imgh, imgw, _ = img.shape
    x,y,w,h = bbox
    x,y = x+w/2-size/2+n, y+h/2-size/2+n
    x, y = max(x,0), max(y,0)
    x, y = min(x,imgw-size), min(y,imgh-size)
    return img[y:y+size, x:x+size, :]

def preprocess(img_dir, bbox_dat):
    """
    Extract pos and neg images by specified bounding boxes
    Return a N*(70*70*c) 2d-array as feature input.
    [args] 
    bbox_dat - name of dat file, assuming it is in img_dir
    img_dir  - contains RGB 3-channeled images
    """
    N = len([f for f in os.listdir(img_dir) if f.endswith('png') and not f.startswith(".")])
    out = np.zeros((N, 3 * patch_dim ** 2))
    print("Start preprocessing...")
    with open(os.path.join(img_dir, bbox_dat), 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        ctr = 0
        for row in reader:
            # print("Jiggling {}/{}".format(ctr+1, N))
            img_path, bbox = row[0], map(int, row[2].split(' '))
            I = misc.imread(os.path.join(img_dir, img_path))
            out[ctr, :] = _jiggle_bbox(I, bbox, patch_dim).reshape(3 * patch_dim**2)
            ctr+=1
    return out

"""Helper to convert brickid to labels 0-9, since we currently only train on a 
10-class data"""
neg_mapping = {3001:0, 3037:1, 2456:2, 2877:3, 3002:4,\
           3003:5,  3004:6, 3005:7, 3010:8, 30363:9}

def _gen_neg_data(mapping, neg_path):
    """[args]mapping - maps brickid to label in training.
    [ret] (X,Y) - X is N*(70*70*c) feature array, Y is N*1 label array"""
    Xacc, Yacc = [], []
    for brickid, label in neg_mapping.iteritems():
        print "Computing for {}".format(brickid)
        img_dir = os.path.join(neg_path, str(brickid) + '_syn')
        X = preprocess(img_dir, 'info.dat') 
        Y = label * np.ones(X.shape[0])
        print X.shape, Y.shape
        Xacc.append(X)
        Yacc.append(Y)
    #stack them together
    return np.vstack(Xacc), np.hstack(Yacc)

def train_single(pos_pkl, neg_pkl):
    """
    Train a single brick classfier against negative data consisting bricks
    of different class.
    """
    highest_acc = 0
    with open(pos_pkl) as f:
        X_pos = pkl.load(f)
        Y_pos = pkl.load(f)
        print X_pos.shape, Y_pos.shape
    with open(neg_pkl) as f:
        X_neg = pkl.load(f)
        Y_neg = pkl.load(f)
        print X_neg.shape, Y_neg.shape
    X, Y = np.vstack([X_pos, X_neg]), np.hstack([Y_pos, Y_neg])
    X, X_val, y, y_val = train_test_split(X, Y)
    enc = OneHotEncoder(categorical_features='all')
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

    model = ShallowNeuralForest(X.shape[1], y.shape[1], regression=False, num_epochs=20)
    def _save_model(fname, model):
        with open(fname, 'w+') as f:
            pkl.dump(model, f, protocol = pkl.HIGHEST_PROTOCOL)
    def on_epoch(epoch, loss, tloss, accur, model):
        global highest_acc
        if accur > highest_acc:
            highest_acc = accur
        #save every ten epoch
        if epoch % 10 == 9:
            _save_model('../models/epoch_{}.pkl'.format(epoch), model)
        print "EPOCH[%3d] accuracy: %.5lf (loss train %.5lf, test %.5lf). Highest accuracy: %.5lf" % (epoch, accur, loss, tloss, highest_acc)

    model.fit(X, y, X_val, y_val, on_epoch=on_epoch, verbose=True)
    _save_model('../models/final.pkl', model)
    

if __name__ == '__main__':
    train_single('../data/pos.pkl', '../data/neg.pkl')

    # img_dir = '../../gen_training/output/3001_antialias_7'
    # out = preprocess(img_dir, 'info.dat')
    # with open( os.path.join('../data', 'pos.pkl'), 'w+') as f:
    #     pkl.dump(out, f, protocol = pkl.HIGHEST_PROTOCOL)
    #     pkl.dump( np.zeros(out.shape[0]), f,protocol = pkl.HIGHEST_PROTOCOL)

    # neg_dir = '../../../neg_images'
    # X,Y = _gen_neg_data(neg_mapping, neg_dir)
    # print X.shape, Y.shape
    # with open( '../data/neg.pkl', 'w+' ) as f:
    #     pkl.dump(X, f, protocol = pkl.HIGHEST_PROTOCOL)
    #     pkl.dump(Y, f, protocol = pkl.HIGHEST_PROTOCOL)