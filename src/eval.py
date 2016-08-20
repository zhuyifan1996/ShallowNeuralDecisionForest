"""script to experiment on how to use the neural forest model"""

from six.moves import cPickle as pkl 
import numpy as np

with open('../example/model.pkl') as f:
	model = pkl.load(f)

print model._x_mean.shape
test = np.zeros((28,28)).reshape(28*28)
print model.predict(test)