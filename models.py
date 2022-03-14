import numpy as np


# Gradient Descent Algorithm for Linear Regression
class LinearRegression:
    def __init__(self, learning_rate=0.0001, stop=1e-6, normalize=True):
        self.m = 0
        self.c = 0
        # self.__data = data
        self.__lr = learning_rate
        self.__stop = stop
        self.__normalize = normalize
        self.__n = None
        self.__mean = None
        self.__std = None
        self.__costs = []
        self.__iterations = []

    def __computeCost(self, y_predict, y):
        loss = np.square(y_predict - y)
        cost = np.sum(loss) / (2 * self.__n)
        return cost

    def __optimize(self, x, y):
        y_predict = np.dot(x, self.m) + self.c
        dm = np.dot(x, (y_predict - y)) / self.__n
        dc = np.sum(y_predict - y) / self.__n
        self.m = self.m - self.__lr * dm
        self.c = self.c - self.__lr * dc

    def __normalizeX(self, x):
        return (x - self.__mean) / self.__std

    def fit(self, x, y):
        if self.__normalize:
            self.__mean, self.__std = x.mean(axis=0), x.std(axis=0)
            x = self.__normalizeX(x)

            self.__n = len(x)
            last_cost, i = float('inf'), 0
            while True:
                y_predict = np.dot(x, self.m) + self.c
                cost = self.__computeCost(y_predict, y)
                self.__optimize(x, y)
                if last_cost - cost < self.__stop:
                    break
                else:
                    last_cost, i = cost, i + 1
                    self.__costs.append(cost)
                    self.__iterations.append(i)

    def predict(self, x):
        if self.__normalize:
            x = self.__normalizeX(x)
        return np.dot(x, self.m) + self.c

    # def future(self, days):
    # self.__data['Future'] = self.__data['Open'].shift(-days)    # Open price after n days
    # new_data = self.__data[['Open', 'Future']]
    # future = new_data

    def score(self, x, y):
        return 1 - (np.sum(((y - self.predict(x)) ** 2)) / np.sum((y - np.mean(y)) ** 2))
