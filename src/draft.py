import numpy as np
import matplotlib.pyplot as plt

def DataLoader(path):
    X, Y = [], []
    for line in open(path, "r", encoding="utf-8"):
        data = list(map(float, line.strip().split(",")))
        X.append(data[:-1])
        Y.append(data[-1])
    
    X = np.array(X)
    Y = np.array(Y)

    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return X, Y

class LinearRegression(object):
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.w = np.random.randn(n_feature, 1)
        self.b = 0
    
    def forward(self, X):
        return np.dot(X, self.w) + self.b
    
    def loss(self, Z, Y):
        return np.mean((Z - Y) ** 2)
    
    def gradient(self, X, Y):
        Z = self.forward(X)
        w_g = np.mean((Z - Y) * X, axis=0)
        w_g = w_g.reshape([-1, 1])
        b_g = np.mean(Z - Y)
        return w_g, b_g
    
    def update(self, w_g, b_g, alpha=0.01):
        self.w -= alpha * w_g
        self.b -= alpha * b_g
    
    def train(self, X, Y, alpha=0.01, n_iter=1000):
        for epoch in range(n_iter):
            Z = self.forward(X)
            loss = self.loss(Z, Y)
            w_g, b_g = self.gradient(X, Y)
            self.update(w_g, b_g, alpha=alpha)

            if epoch % 100 == 0:
                print(f"epoch:{epoch}, loss:{loss}")
    
X, Y = DataLoader("../data/boston.txt")
print(X.shape, Y.shape)
Y = Y.reshape([-1, 1])

ratio = 0.8
offline = int(ratio * X.shape[0])
train_X = X[:offline]
train_Y = Y[:offline]
test_X = X[offline:]
test_Y = Y[offline:]

model = LinearRegression(n_feature=13)
model.train(X=train_X, Y=train_Y, alpha=0.01, n_iter=1000)

predict = model.forward(test_X)

plt.plot(predict, label="predict")
plt.plot(test_Y, label="test_Y")
plt.grid(True)
plt.legend()
plt.show()