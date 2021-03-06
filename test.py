from net import NeuralNet;
from sklearn.externals import joblib;
import numpy as np;

from scipy.misc import imresize;
from PIL import Image;
from io import BytesIO;
from skimage.morphology import skeletonize;
import base64;

import sys;
# img_base64_before_split=sys.argv[1];
img_base64_before_split='data:image/jpeg;base64,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';
img_base64=img_base64_before_split.split('jpeg;base64,')[1];
img_grayscale = Image.open(BytesIO(base64.b64decode(img_base64))).convert('L');
img=np.array(img_grayscale);
img_resized=imresize(img,(28,28));

inp = np.reshape(img_resized, (784,1));

net_obj = joblib.load('NeuralNet.sav');

outi = net_obj.NNout(inp);
print(outi);

