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
    
    mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(), # one hidden layer with 20 features 
                                           activation='relu',        # rectified linear
                                           learning_rate_init=1e-2,  # learning rate
                                           max_iter=1000,            # number of iterations
                                           early_stopping=True,      # stop training if validation data gets worse
                                           random_state=0)           # random number seed for initialization
    
    
    digits = sklearn.datasets.load_digits()
    
    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        digits.data, 
        digits.target, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0
    )
    
    mlp.fit(x_train, y_train)
    output = mlp.predict(x_test)
    
    accuracy = np.mean(output == y_test)
    
    print(accuracy)
    
    confusion = np.zeros((10,10))
    for i in range(len(output)):
        confusion[output[i], y_test[i]] += 1
    print(confusion)
    
    discussion = """
        Since there are no hidden layers, I would expect the MLP to just be a single perceptron.
        However, the network performs quite well for a single perceptron..
        
        Since having an MLP with no hidden layers doesn't make a ton of sense, and since the 
        accuracy is so high, I suspect that there may actually be some default number of layers
        going on in the initializer for the MLP.
    
    """
    
    print(discussion)
    
    
    
    
    
    




