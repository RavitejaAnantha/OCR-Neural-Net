import gzip
import pickle
import numpy as np;
import matplotlib.cm as cm
import matplotlib.pyplot as plt


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f);

train_x, train_y = train_set;
a = np.array(train_x[0]);
print(a.shape);

plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r);
plt.show();