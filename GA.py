import numpy as np
from math import exp, pi, sqrt
import pandas as pd


def standardize(X, mean, std):
    X_s = (X - mean)/std
    return X_s


class GA:
    # ID is identifier, it is 1 for GDA and 2 for NBGDA (Naive Bayessian Gaussian discriminant analysis)
    def __init__(self, ID):
        # Initialize parameters
        self.ID = ID
        self.Phi_0 = 0
        self.Phi_1 = 0
        self.Phi_2 = 0 # This doesn't need to be calculated
        self.mi_0 = 0
        self.mi_1 = 0
        self.mi_2 = 0
        self.sigma_0 = 0
        self.sigma_1 = 0
        self.sigma_2 = 0

    def fit(self, X, y):
        # X is mxn matrix
        m = np.size(X, axis=0)
        n = np.size(X, axis=1)
        y = y[:,np.newaxis]

        # Calculate parameters:
        self.Phi_0 = (np.where(y == 0, 1, 0)).sum()/m
        self.Phi_1 = (np.where(y == 1, 1, 0)).sum()/m
        self.Phi_2 = (np.where(y == 2, 1, 0)).sum()/m

        self.mi_0 = np.sum(np.where(y == 0, X, 0), axis=0)/np.where(y == 0, 1, 0).sum()
        self.mi_1 = np.sum(np.where(y == 1, X, 0), axis=0)/np.where(y == 1, 1, 0).sum()
        self.mi_2 = np.sum(np.where(y == 2, X, 0), axis=0)/np.where(y == 2, 1, 0).sum()

        x_0 = X[np.argwhere(y == 0)]
        x_1 = X[np.argwhere(y == 1)]
        x_2 = X[np.argwhere(y == 2)]

        #print(x_0)

        for i in range(0, np.size(x_0, axis=0)):
            self.sigma_0 = self.sigma_0 + (x_0[i] - self.mi_0).T@(x_0[i] - self.mi_0)
        self.sigma_0 = self.sigma_0/np.size(x_0, axis=0)

        for i in range(0, np.size(x_1, axis=0)):
            self.sigma_1 = self.sigma_1 + (x_1[i] - self.mi_1).T@(x_1[i] - self.mi_1)
        self.sigma_1 = self.sigma_1/np.size(x_1, axis=0)

        for i in range(0, np.size(x_2, axis=0)):
            self.sigma_2 = self.sigma_2 + (x_2[i] - self.mi_2).T@(x_2[i] - self.mi_2)
        self.sigma_2 = self.sigma_2/np.size(x_2, axis=0)

        if self.ID == 2:
            self.sigma_0 = self.sigma_0*np.eye(n)
            self.sigma_1 = self.sigma_1*np.eye(n)
            self.sigma_2 = self.sigma_2*np.eye(n)

    def predict(self, x):
        # Calculate all needed probabilities for given x
        p_x_y_0 = exp(-((x.T - self.mi_0.T).T@(np.linalg.inv(self.sigma_0))@(x.T-self.mi_0.T))/2)/(((2*pi)**(5/2))*(sqrt(np.linalg.det(self.sigma_0))))
        p_x_y_1 = exp(-((x.T - self.mi_1.T).T@(np.linalg.inv(self.sigma_1))@(x.T-self.mi_1.T))/2)/(((2*pi)**(5/2))*(sqrt(np.linalg.det(self.sigma_1))))
        p_x_y_2 = exp(-((x.T - self.mi_2.T).T@(np.linalg.inv(self.sigma_2))@(x.T-self.mi_2.T))/2)/(((2*pi)**(5/2))*(sqrt(np.linalg.det(self.sigma_2))))

        s_pr = p_x_y_0*self.Phi_0 + p_x_y_1*self.Phi_1 + p_x_y_2*self.Phi_2
        Pr = np.zeros(3)
        Pr[0] = p_x_y_0*self.Phi_0/s_pr
        Pr[1] = p_x_y_1*self.Phi_1/s_pr
        Pr[2] = p_x_y_2*self.Phi_2/s_pr

        y_pred = np.argmax(Pr)
        return Pr, y_pred


df = pd.read_csv(r'C:\Users\Suzana\Desktop\MU_dom_2_Suzana_Mandic\multiclass_data.csv', header=None)
data = df.values
Acc = np.zeros(20)
for i in range(1, 21):
    # Shuffle data
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

    gda_classifier = GA(2)
    gda_classifier.fit(X_train, y_train)
    y_pred = np.zeros((len(y_test), 1))
    for j in range(0, np.size(X_test, axis=0)):
        y_pred[j] = gda_classifier.predict(X_test[j, :])[1]
    y_test = y_test[:, np.newaxis]
    # Calculate accuracy
    Acc[i-1] = np.where(y_pred == y_test, 1, 0).sum()/np.size(y_test, axis=0)
print(Acc.sum()/20)

