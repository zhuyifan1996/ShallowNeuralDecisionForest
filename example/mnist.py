import sys
import os.path
sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from neuralforest import ShallowNeuralForest

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

mnist = fetch_mldata('MNIST original')

highest_acc = 0
def on_epoch(epoch, loss, tloss, accur, model):
    global highest_acc
    if accur > highest_acc:
        highest_acc = accur
    print "EPOCH[%3d] accuracy: %.5lf (loss train %.5lf, test %.5lf). Highest accuracy: %.5lf" % (epoch, accur, loss, tloss, highest_acc)

X, X_val, y, y_val = train_test_split(mnist.data, mnist.target)
print X.shape, X[:2]
print y.shape, y[:10]

enc = OneHotEncoder(categorical_features='all')
y = enc.fit_transform(y.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
print y.shape, y[:10]

model = ShallowNeuralForest(X.shape[1], y.shape[1], regression=False, num_epochs=1)
model.fit(X, y, X_val, y_val, on_epoch=on_epoch, verbose=True)

# Z = np.zeros((28,28)).reshape(784)
print model.predict(X_val)

# from six.moves import cPickle as pkl 
# with open('model.pkl', 'w+') as f:
# 	pkl.dump(model, f, protocol = pkl.HIGHEST_PROTOCOL)

