import numpy as np;
import sys;
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
                # print('NN Output: ', outi);
        return outi;

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
        print('Output of NN: ', self.feedforward(x));
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data];
        return sum(int(x == y) for (x, y) in test_results);

    def cost_derivative(self, output_activations, y):
        return (output_activations-y);

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z));

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z));