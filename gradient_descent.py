import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_model(X, y, alpha, max_epoch):
    '''
    X: input features
    y: responses
    alpha: learning rate
    max_epoch: maximum epochs

    returns w, hist_loss
    '''
    np.random.seed(2)

    m = X.shape[1]
    w = np.random.rand(m,)  # initialise w
    hist_loss = []

    for i in range(max_epoch):
        yhat = predict(w, X) 
        loss = loss_fn(y, yhat) 
        hist_loss.append(loss)
        # X.T.dot(yhat-y) is the partial derivative of the loss function
        w -= alpha / m * X.T.dot(yhat-y)
        print(f'epoch {i+1}: {loss:.3f}')

    return w, hist_loss

def predict(w, X):
    '''
    w: weights
    X: input features
    '''
    return np.dot(X, w)

def loss_fn(y, yhat):
    '''
    y: responses
    yhat: predicted values
    '''
    # Loss function is squared loss
    return np.mean(np.square(yhat - y)) / 2

df = pd.read_csv('assignment1_dataset.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Introduce a trainable parameter x_0 = 1
x_0 = np.ones((X.shape[0], 1))
X = np.concatenate((x_0, X), axis=1)

LR = 0.01
N_ITERS = 300

w, hist_loss = train_model(X, y, LR, N_ITERS)

print(f'w = {w}')

plt.figure(figsize=(8,10))
plt.plot(range(1, N_ITERS+1), hist_loss)
plt.title('Squared loss against epoch')
plt.xlabel('epoch')
plt.ylabel('Squared loss')
plt.show()
