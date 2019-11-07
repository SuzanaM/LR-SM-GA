import numpy as np
import pandas as pd
from math import exp, log
import matplotlib.pyplot as plt

# Define function for standardization


def standardize(X, mean, std):
    X_s = (X - mean)/std
    return X_s

# Define Classifier class


class Log_Classifier:

    def __init__(self, ID):
        self.ID = ID
        self.params = np.zeros((0, 0))

    # Define hypothesis
    def h(self, z):
        # Check if it can work with vectors
        sigmoid = 1 / (1 + exp(-z))
        return sigmoid

    # Define conditional probability
    '''
    def cond_prob(self, x, y):
        # If y is 1, where 1 defines right class, then cond_prob is hypothesis(x)
        p = (self.h(x) ** y) * ((1 - self.h(x)) ** (1 - y))
        return p
    '''

    # Define cost function
    def cost(self, X, y):
        #CP_vector = np.zeros((len(Y), 1))
        #for i in range(0, len(Y)):
        #    TTX = self.params.T @ X[i, :].T
        #    CP_vector[i] = self.cond_prob(TTX, Y[i])
        #J = -((np.log(CP_vector)).sum())
        J = 0
        for i in range(0, np.size(X, axis=0)):
            J = J + (y[i]*log(self.h(self.params.T@X[i, :].T)) + (1 - y[i])*log(1 - self.h(self.params.T@X[i, :].T)))
        return J

    def fit(self, X, y, alpha):
        # Number of training examples
        m = np.size(X, axis=0)
        # Number of features
        n = np.size(X, axis=1)
        # x_0 = 1
        X = np.hstack((np.ones((m, 1)), X))
        y = y[:, np.newaxis]
        # Number of iterations for BGD
        n_iter = 1500
        # Initialize vector of cost function through iterations
        J_history = np.zeros((n_iter, 1))
        # Depending on ID, this function fits classifier to be classifier for class 0, class 1 or class 2
        if self.ID == 0:
            y = np.where(y == 0, 1, 0)
        elif self.ID == 1:
            y = np.where(y == 1, 1, 0)
        else:  # ID == 2
            y = np.where(y == 2, 1, 0)

        # Implementation of batch gradient descent
        # Initialize parameters to 0
        '''
        self.params = np.zeros((n+1, 1))
        for i in range(0, n_iter):
            for j in range(0, n+1):
                derivative = 0
                for k in range(0, m):
                    derivative = derivative + ((self.h(self.params.T @ X[k, :].T) - y[k])*X[k, j])
                self.params[j] = self.params[j] - alpha*derivative
            J_history[i] = self.cost(X, y)
        '''
        self.params = np.zeros((n + 1, 1))
        for i in range(0, n_iter):
            derivative = 0
            for k in range(0, m):
                derivative = derivative + ((self.h(self.params.T @ X[k, :].T) - y[k])*X[k, :].T)
            # Derivative must be 2d array to be able to transpose
            derivative = derivative[:, np.newaxis]
            self.params = self.params - alpha * derivative
            J_history[i] = -self.cost(X, y)

        return J_history

    def predict(self, x):
        x = np.hstack((1, x))
        TTX = self.params.T @ x.T
        hypothesis = self.h(TTX)
        return hypothesis

    def get_params(self):
        return self.params


# Read data from csv
df = pd.read_csv(r'C:\Users\Suzana\Desktop\MU_dom_2_Suzana_Mandic\multiclass_data.csv', header=None)
data = df.values

'''
# Check the ratio of classes
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
    np.random.seed(i)
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
    # Create classifier for class 0
    classifier_0 = Log_Classifier(ID=0)
    J_history_0 = classifier_0.fit(X=X_train, y=y_train, alpha=0.01)

    #plt.plot(np.linspace(0, 1500, 1500), J_history_0)
    #plt.xlabel('#iteracija')
    #plt.ylabel('J')
    #plt.grid()
    #plt.show()

    # Create classifier for class 1
    classifier_1 = Log_Classifier(ID=1)
    J_history_1 = classifier_1.fit(X=X_train, y=y_train, alpha=0.01)

    # Create classifier for class 2
    classifier_2 = Log_Classifier(ID=2)
    J_history_2 = classifier_2.fit(X=X_train, y=y_train, alpha=0.01)

    y_pred = np.zeros((len(y_test), 1))
    for j in range(0, len(y_test)):
        pred_0 = classifier_0.predict(X_test[j, :])
        pred_1 = classifier_1.predict(X_test[j, :])
        pred_2 = classifier_2.predict(X_test[j, :])
        if (pred_0 > pred_1 and pred_0 > pred_2):
            y_pred[j] = 0
        elif (pred_1 > pred_0 and pred_1 > pred_2):
            y_pred[j] = 1
        elif (pred_2 > pred_0 and pred_2 > pred_1):
            y_pred[j] = 2

    y_test = y_test[:, np.newaxis]
    # Calculate accuracy
    Acc[i-1] = np.where(y_pred == y_test, 1, 0).sum()/np.size(y_test, axis=0)
print(Acc.sum()/20)





