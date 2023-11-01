# Student ID: 20838709

import sklearn
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    x, y = sklearn.datasets.make_circles(n_samples=100, 
                                         shuffle=True, 
                                         noise=0.1, 
                                         random_state=0, 
                                         factor=0.3)

    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        x, 
        y, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0
    )
    

    
    #1D) [DONE]
    
    run_len = 10
    avgd_RMSEs = []
    feature_cnts = np.arange(5, 50, 5)
    
    for feature_cnt in feature_cnts:
        
        RMSEs = []
        
        for i in range(run_len):
            
            x_train, x_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(
                x, 
                y, 
                test_size=0.2, 
                shuffle=True, 
                random_state=i
            )
        
            mlp = MLPClassifier(
                hidden_layer_sizes=(feature_cnt,), #one hidden layer with 20 features
                activation='relu', #rectified linear
                learning_rate_init=1e-2, #learning rate
                max_iter=1000, # number of iterations
                early_stopping=True, #stop training if validation data gets worse
                random_state=i#random number seed for initialization
            ) 
            
            mlp.fit(x_train, y_train)
            
            output = mlp.predict(x_test)
            
            diff = np.array(y_test - output)
            
            rmse = np.sqrt(np.mean(np.square(diff)))
            
            RMSEs.append(rmse)
        
        avgd_RMSEs.append(sum(RMSEs) / run_len)
        
    plt.plot(feature_cnts, avgd_RMSEs)
    plt.title("Number of features vs RMSE")
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.ylim([0, 1])
    plt.grid()
    #plt.show()
    

    
    




