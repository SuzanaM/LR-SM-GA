import numpy as np
import random
from math import exp
import pandas as pd
import matplotlib.pyplot as plt


def standardize(X, mean, std):
    X_s = (X - mean)/std
    return X_s

# Create class for softmax


class Softmax:

    def __init__(self, k, n):
        # k indicates number of classes
        self.k = k
        # n indicates number of features
        self.n = n
        self.params = np.zeros((self.n+1, self.k))

    def lin_comb(self, l, x):
        lin_comb_l = exp(self.params[:, l].T@x)

        lin_comb_o = 0

        for i in range(0, self.k):
            lin_comb_o = lin_comb_o + exp(self.params[:, i].T@x)

        a = lin_comb_l/lin_comb_o

        return a

    def cost(self, X, y):
        J = 0
        for i in range(0, np.size(X, axis=0)):
            l = int(y[i])
            s = 0
            for j in range(0, self.k):
                s = s + exp(self.params[:, j].T@X[i, :].T)
            sg = self.params[:, l].T@X[i, :].T - np.log(s)
            J = J + sg
        return J

    def fit(self, X, y, alpha, n_mb):
        m = np.size(X, axis=0)
        X = np.hstack((np.ones((m, 1)), X))
        y = y[:, np.newaxis]
        n_iterations = 1500
        J_history = np.zeros((n_iterations, 1))
        # Do mini-batch gradient descent
        indices = list(range(0, m))
        for i in range(0, n_iterations):
            # Randomly take 2**something examples
            mb = random.sample(indices, 2**n_mb)
            for l in range(0, self.k - 1):
                derivative = 0
                for j in mb:  # Through mini batch we find derivative which will be added to update parameters
                    derivative = derivative + (np.where(y[j] == l, 1, 0) - self.lin_comb(l, X[j, :].T))*X[j, :].T
                self.params[:, l] = self.params[:, l] + alpha*derivative
            J_history[i] = self.cost(X, y)
        return J_history

    def predict(self, x):
        x = np.hstack((1, x))
        h = np.zeros((self.k-1, 1))
        for i in range(0, self.k-1):
            s = 0
            for j in range(0, self.k):
                s = s + exp(self.params[:, j].T@x.T)
            h[i] = exp(self.params[:, i].T@x.T)/s
        return h

    def get_params(self):
        return self.params


# Read data from csv
df = pd.read_csv(r'C:\Users\Suzana\Desktop\MU_dom_2_Suzana_Mandic\multiclass_data.csv', header=None)
data = df.values

'''
# Check the number of classes examples
n_0 = (np.where(data[:, 5] == 0, 1, 0)).sum()
n_1 = (np.where(data[:, 5] == 1, 1, 0)).sum()
n_2 = (np.where(data[:, 5] == 2, 1, 0)).sum()

print(n_0)
print(n_1)
print(n_2)
'''
# Shuffle data
Acc = np.zeros(20)
for i in range(1, 21):
    np.random.seed(15)
    np.random.shuffle(data)

    # Split data into train and test set, validation set isn't needed
    ind = int(0.7*np.size(data, axis=0))
    train = data[0:ind, :]
    test = data[ind:, :]
    X_train = train[:, 0:5]
    y_train = train[:, 5]
    X_test = test[:, 0:5]
    y_test = test[:, 5]

    # Standardize data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = standardize(X_train, mean, std)
    X_test = standardize(X_test, mean, std)

    '''
    # Check if standardization is good
    print(np.mean(X_test, axis=0))
    print(np.std(X_test, axis=0))
    '''

    softmax_classifier = Softmax(k=3, n=np.size(X_train, axis=1))
    J_h = softmax_classifier.fit(X_train, y_train, alpha=0.01, n_mb=4)
    #plt.plot(np.linspace(0, 1500, 1500), J_h)
    #plt.xlabel('#iteracija')
    #plt.ylabel('J')
    #plt.grid()
    #plt.show()

    y_pred = np.zeros((len(y_test), 1))
    for j in range(0, len(y_test)):
        pred = softmax_classifier.predict(X_test[j, :])
        if pred[0] > pred[1] and pred[0] > (1 - (pred[0] + pred[1])):
            y_pred[j] = 0
        elif pred[1] > pred[0] and pred[1] > (1 - (pred[0] + pred[1])):
            y_pred[j] = 1
        elif (1 - (pred[1] + pred[0])) > pred[0] and (1 - (pred[1] + pred[0])) > pred[1]:
            y_pred[j] = 2

    y_test = y_test[:, np.newaxis]
    # Calculate accuracy
    Acc[i-1] = np.where(y_pred == y_test, 1, 0).sum() / np.size(y_test, axis=0)
print(Acc.sum()/20)








