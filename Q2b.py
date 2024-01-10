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
    
    num_features = np.arange(5, 51, 5)
    accuracies = []
    run_len = 10
    
    digits = sklearn.datasets.load_digits()
    
    # Set the dataset once, so each run we are only changing the classifier, and can
    # Fairly compare one classifier to the next, on the SAME dataset.
    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        digits.data, 
        digits.target, 
        test_size=0.2, 
        shuffle=True, 
        random_state=10
    )
    
    j = 0
    for feature_cnt in num_features:
        
        accuracies_per_run = []
        for k in range(run_len):    
    
            mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(feature_cnt,), # one hidden layer with 20 features 
                                                   activation='relu',        # rectified linear
                                                   learning_rate_init=1e-2,  # learning rate
                                                   max_iter=1000,            # number of iterations
                                                   early_stopping=True,      # stop training if validation data gets worse
                                                   random_state=k)           # random number seed for initialization
            
            mlp.fit(x_train, y_train)
            output = mlp.predict(x_test)
            
            accuracy = np.mean(output == y_test) * 100
            accuracies_per_run.append(accuracy)
            
            print("Num feautres:", feature_cnt, "Accuracy:", accuracy)
            
        accuracies.append(np.mean(accuracies_per_run))
            
        j += 1
        
plt.plot(num_features, accuracies)
plt.xlabel("Number of features")
plt.ylabel("% Accuracy")
plt.title("% Accruacy vs number of features")
plt.grid()
plt.show()

    
    
    




