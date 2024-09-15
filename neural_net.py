from typing import List
import numpy as np

from operations import *

class NeuralNetwork():

    def __init__(self, n_features: int, layer_sizes: List[int], activations: List[Activation], loss: Loss,
                 learning_rate: float=0.01, W_init: List[np.ndarray]=None):

        sizes = [n_features] + layer_sizes
        if W_init:
            assert all([W_init[i].shape == (sizes[i] + 1, sizes[i+1]) for i in range(len(layer_sizes))]), \
                "Specified sizes for layers do not match sizes of layers in W_init"
        assert len(activations) == len(layer_sizes), \
            "Number of sizes for layers provided does not equal the number of activations provided"

        self.n_layers = len(layer_sizes)
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.W = []
        for i in range(self.n_layers):
            if W_init:
                self.W.append(W_init[i])
            else:
                rand_weights = np.random.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i])
                biases = np.zeros((1, sizes[i+1]))
                self.W.append(np.concatenate([biases, rand_weights], axis=0))

    def forward_pass(self, X) -> tuple[List[np.ndarray], List[np.ndarray]]:
        A_vals = []
        Z_vals = []
        X = np.c_[np.ones(len(X)), X] 
        a = np.matmul(X, self.W[0])
        A_vals.append(a)
        z = self.activations[0].value(a)
        Z_vals.append(z)
        i = 1
        while(i < self.n_layers):
            tmp = np.c_[np.ones(len(X)), Z_vals[i-1]] 
            a = np.matmul(tmp,self.W[i])
            A_vals.append(a)
            z = self.activations[i].value(a)
            Z_vals.append(z)
            i += 1
        return A_vals, Z_vals

    def backward_pass(self, A_vals, dLdyhat) -> List[np.ndarray]:
        deltas = [None] * self.n_layers
        d = np.multiply(dLdyhat, self.activations[self.n_layers-1].derivative(A_vals[self.n_layers-1]))
        deltas[self.n_layers - 1] = d 
        for i in range(self.n_layers-2, -1, -1):
            newW = np.delete(self.W[i+1], (0), axis=0)
            d = np.multiply(np.matmul(deltas[i+1], np.transpose(newW)), self.activations[i].derivative(A_vals[i]))
            deltas[i] = d
        return deltas

    def update_weights(self, X, Z_vals, deltas) -> List[np.ndarray]:
        W = [None] * self.n_layers
        newX = np.c_[np.ones(len(X)), X] 
        e = np.matmul(np.transpose(newX), deltas[0])
        W[0] = self.W[0] - self.learning_rate * e
        i = 1
        while i < self.n_layers:
            newZ = np.c_[np.ones(len(Z_vals[i-1])), Z_vals[i-1]] 
            e = np.matmul(np.transpose(newZ), deltas[i])
            W[i] = np.subtract(self.W[i], self.learning_rate * e)
            i += 1

        return W

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> (List[np.ndarray], List[float]):
        epoch_losses = []
        for epoch in range(epochs):
            A_vals, Z_vals = self.forward_pass(X)
            y_hat = Z_vals[-1] 
            L = self.loss.value(y_hat, y) 
            print("Epoch {}/{}: Loss={}".format(epoch, epochs, L))
            epoch_losses.append(L)  

            dLdyhat = self.loss.derivative(y_hat, y) 
            deltas = self.backward_pass(A_vals, dLdyhat) 
            self.W = self.update_weights(X, Z_vals, deltas) 

        return self.W, epoch_losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric) -> float:
        A_vals, Z_vals = self.forward_pass(X) 
        y_hat = Z_vals[-1]
        metric_value = metric(y_hat, y)
        return metric_value

