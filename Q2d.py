# Student ID: 20838709

import sklearn
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.datasets


if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    digits = sklearn.datasets.load_digits()
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        digits.data, digits.target, test_size=0.2, shuffle=True, random_state=0,
    )
    
    

    
    
    
    
    
    




