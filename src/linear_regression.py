import numpy as np

def DataLoader(path):
    X, Y = [], []
    for line in open(path, "r", encoding="utf-8"):
        data = list(map(float, line.strip().split(",")))
        X.append(data[:-1])
        Y.append(data[-1])
    X = np.array(X)
    Y = np.array(Y)

    # 数据归一化
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return X, Y

class LinearRegression(object):
    def __init__(self, feature_num):   # 特征维度
        np.random.seed(123)    
        self.w = np.random.randn(feature_num, 1)    # 正态分布初始化权重向量
        self.b = 0
    
    def forward(self, X):
        return np.dot(X, self.w) + self.b
    
    def loss(self, Z, Y):
        return np.mean((Z - Y) ** 2)
    
    def gradient(self, X, Y):
        Z = self.forward(X)                 # 前向运算得到预测值
        w_g = np.mean((Z - Y) * X, axis=0)  # 广播运算+求平均值得到梯度
        w_g = w_g.reshape([-1, 1])          # 梯度要写成列向量
        b_g = np.mean(Z - Y)
        return w_g, b_g
    
    def update(self, w_g, b_g, alpha=0.01):
        self.w -= alpha * w_g
        self.b -= alpha * b_g
    
    def train(self, X, Y, n_iter, alpha=0.01):
        for i in range(n_iter):
            Z = self.forward(X)
            loss = self.loss(Z, Y)
            w_g, b_g = self.gradient(X, Y)
            self.update(w_g, b_g)

            if (i + 1) % 100 == 0:
                print("iter {}, loss {}".format(i, loss))
        
    def __call__(self, X, Y, n_iter, alpha=0.01):
        self.train(X, Y, n_iter, alpha)


X, Y = DataLoader("./data/boston.txt")

LinearRegressionModel = LinearRegression(feature_num=13)
LinearRegressionModel(X, Y.reshape([-1, 1]), 1000, 0.01)