import numpy as np


def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(x_shift)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    N = y.shape[0]
    correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-8)
    return np.sum(correct_logprobs) / N

def compute_accuracy(preds, y):
    return np.mean(preds == y)



class NeuralNet:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
            h1, h2 = hidden_sizes
            self.params = {}


            if activation == 'relu':
                self.activation = relu
                self.activation_deriv = relu_derivative
                scale = lambda fan_in: np.sqrt(2. / fan_in)
            elif activation == 'sigmoid':
                self.activation = sigmoid
                self.activation_deriv = sigmoid_derivative
                scale = lambda fan_in: np.sqrt(1. / fan_in)
            else:
                raise ValueError("Unsupported activation")


            self.params['W1'] = scale(input_size) * np.random.randn(input_size, h1)
            self.params['b1'] = np.zeros((1, h1))

            self.params['W2'] = scale(h1) * np.random.randn(h1, h2)
            self.params['b2'] = np.zeros((1, h2))

            self.params['W3'] = scale(h2) * np.random.randn(h2, output_size)
            self.params['b3'] = np.zeros((1, output_size))

    def forward(self, X):
        self.cache = {}
        self.cache['X'] = X

        Z1 = X @ self.params['W1'] + self.params['b1']
        A1 = self.activation(Z1)

        Z2 = A1 @ self.params['W2'] + self.params['b2']
        A2 = self.activation(Z2)

        Z3 = A2 @ self.params['W3'] + self.params['b3']
        A3 = softmax(Z3)


        self.cache.update({'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3})
        return A3

    def backward(self, y, reg=0.0):
        grads = {}
        m = y.shape[0]
        X = self.cache['X']
        A1, A2, A3 = self.cache['A1'], self.cache['A2'], self.cache['A3']


        y_one_hot = np.zeros_like(A3)
        y_one_hot[np.arange(m), y] = 1


        dZ3 = (A3 - y_one_hot) / m
        grads['W3'] = A2.T @ dZ3 + reg * self.params['W3']
        grads['b3'] = np.sum(dZ3, axis=0, keepdims=True)


        dA2 = dZ3 @ self.params['W3'].T
        dZ2 = dA2 * self.activation_deriv(self.cache['Z2'])
        grads['W2'] = A1.T @ dZ2 + reg * self.params['W2']
        grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)


        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * self.activation_deriv(self.cache['Z1'])
        grads['W1'] = X.T @ dZ1 + reg * self.params['W1']
        grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)

        self.grads = grads

    def update(self, learning_rate):
        for param in self.params:
            self.params[param] -= learning_rate * self.grads[param]

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def compute_loss_and_accuracy(self, X, y, reg=0.0):
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, y)
        loss += 0.5 * reg * sum(np.sum(self.params[f"W{i}"] ** 2) for i in range(1, 4))
        acc = compute_accuracy(np.argmax(probs, axis=1), y)
        return loss, acc
