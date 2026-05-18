import numpy as np

class BackpropNetwork:
    def __init__(self, n_input, n_h1, n_h2, lr=0.01):
        self.lr = lr
        np.random.seed(42)
        self.W1 = np.random.randn(n_input, n_h1) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_h1))
        self.W2 = np.random.randn(n_h1,   n_h2) * np.sqrt(2.0 / n_h1)
        self.b2 = np.zeros((1, n_h2))
        self.W3 = np.random.randn(n_h2,   1)    * np.sqrt(2.0 / n_h2)
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X  @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = np.maximum(0, self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        return self.Z3

    def backward(self, X, y_true, y_pred):
        n  = len(y_true)
        dL = 2 * (y_pred - y_true.reshape(-1, 1)) / n

        dW3 = self.A2.T @ dL
        db3 = np.sum(dL, axis=0, keepdims=True)

        dA2 = dL @ self.W3.T
        dZ2 = dA2 * (self.Z2 > 0).astype(float)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0).astype(float)
        dW1 = X.T  @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return self.forward(X).flatten()