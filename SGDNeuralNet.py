import numpy as np;
import cPickle;
import gzip;
import random;
from sklearn.externals import joblib;
import sys

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb');
    training_data, validation_data, test_data = cPickle.load(f);
    f.close();
    return (training_data, validation_data, test_data); #Ravi isn't using validation_data since he is only building one model and that is SGD

def load_data_wrapper():
	print('*****:Loading MNIST Data*****');
	print('...');
	tr_d, va_d, te_d = load_data();
	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]];
	training_results = [vectorized_result(y) for y in tr_d[1]];
	training_data = zip(training_inputs, training_results);
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]];
	validation_data = zip(validation_inputs, va_d[1]);
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]];
	test_data = zip(test_inputs, te_d[1]);
	print('Data ready for your Cool Machine Learning Stuff!!!');
	return (training_data, validation_data, test_data);

def vectorized_result(j):
    e = np.zeros((10, 1));
    e[j] = 1.0;
    return e;

class NeuralNet(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes);
        self.sizes = sizes;
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]];
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])];

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b);
        return a;

    def NNout(self, a):
        for b, w in zip(self.biases, self.weights):
            flag = False;
            if a.shape[0] == 30:
                flag = True;
                outi = 0;
            a = sigmoid(np.dot(w, a)+b);
            if flag:
                maxi = -sys.maxint;
                for i in xrange(len(a)):
                    if a[i]>maxi:
                        maxi = a[i];
                        outi = i;
                print('NN Output: ', outi);
        return a;

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: 
            n_test = len(test_data);
        n = len(training_data);
        for j in xrange(epochs):
            random.shuffle(training_data);
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)];
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta);
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test));
            else:
                print("Epoch {0} complete".format(j));

    def update_mini_batch(self, mini_batch, eta):
        derv_b = [np.zeros(b.shape) for b in self.biases];
        derv_w = [np.zeros(w.shape) for w in self.weights];
        for x, y in mini_batch:
            delta_derv_b, delta_derv_w = self.backprop(x, y);
            derv_b = [nb+dnb for nb, dnb in zip(derv_b, delta_derv_b)];
            derv_w = [nw+dnw for nw, dnw in zip(derv_w, delta_derv_w)];
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, derv_w)];
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, derv_b)];

    def backprop(self, x, y):
        derv_b = [np.zeros(b.shape) for b in self.biases];
        derv_w = [np.zeros(w.shape) for w in self.weights];
        # feedforward
        activation = x;
        activations = [x]; # list to store all the activations, layer by layer
        zs = []; # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b;
            zs.append(z);
            activation = sigmoid(z);
            activations.append(activation);

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]);
        derv_b[-1] = delta;
        derv_w[-1] = np.dot(delta, activations[-2].transpose());

        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in xrange(2, self.num_layers):
            z = zs[-l];
            sp = sigmoid_prime(z);
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp;
            derv_b[-l] = delta;
            derv_w[-l] = np.dot(delta, activations[-l-1].transpose());
        return (derv_b, derv_w);

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data];
        return sum(int(x == y) for (x, y) in test_results);

    def cost_derivative(self, output_activations, y):
        return (output_activations-y);

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z));

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z));

def main():
	training_data, validation, test_data = load_data_wrapper();
	print('The Neural Net can only have THREE LAYERS, one Input Layer, one Hidden Layer, and one Output Layer');
	inpL = input('Enter the no. of Neurons at the Input Layer: ');
	hidL = input('Enter the no. of Neurons at the Hidden Layer: ');
	outL = input('Enter the no. of Neurons at the Output Layer: ');
	print('Creating your Neural Net...');
	net = NeuralNet([inpL, hidL, outL]);
	print('To run SGD we need some config details');
	ep = input('Enter the no. of Epochs: ');
	minBatch = input('Enter the size of a Mini-Batch: ');
	eta = input('Enter the Learning Rate: ');
	print('Running SGD...');
        net.SGD(training_data, ep, minBatch, eta, test_data=test_data);
        filename = 'NeuralNet.sav';
        joblib.dump(net, filename);

main();
