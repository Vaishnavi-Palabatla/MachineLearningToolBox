import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def hypothesis(X,theta):
    return np.dot(X,theta)

def computeCost(X,y,theta):
    m=len(y)
    # predictions=X.dot(theta)
    predictions = hypothesis(X,theta)
    predictions = np.sum(predictions,axis = 1)
    square_err=(predictions - y)**2
    return 1/(2*m) * np.sum(square_err)

def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    
    m=len(y)
    J_history=[]
    # print("x #########",X.shape)
    # print("theta ###########",theta.shape)

    for i in range(num_iters):
        predictions = X.dot(theta)
        # print("G@@@@@@D predcitions = ",predictions.shape)
        error = np.dot(X.transpose(),(predictions -y))
        # print("@@@@@@@@@@@@@@@@@@@@@@",error)
        descent=alpha * 1/m * error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    
    return theta, J_history

def predict(x,theta):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    
    predictions= np.squeeze(np.dot(x,theta))
    # print("predictions = ",predictions)
    return predictions

def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    
    X_norm = (X - mean)/std
    
    return X_norm , mean , std