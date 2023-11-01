# Student ID: 20838709

import sklearn
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    x, y = sklearn.datasets.make_moons(n_samples=500, noise=0.05, random_state=0)

    #1H)
    

    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        x, 
        y, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0
    )
    
    #1A) [DONE]
    mlp = MLPClassifier(
        hidden_layer_sizes=(20,), #one hidden layer with 20 features
        activation='relu', #rectified linear
        learning_rate_init=1e-2, #learning rate
        max_iter=1000, # number of iterations
        early_stopping=True, #stop training if validation data gets worse
        random_state=0#random number seed for initialization
    ) 
    
    mlp.fit(x_train, y_train)
    
    output = mlp.predict(x_test)
    
    diff = np.array(y_test - output)
    
    rmse = np.sqrt(np.mean(np.square(diff)))
    
    print("1H RMSE (no scaler): ", rmse)
    
    extent = (-2, 2, -2, 2)
    G = 200
    XX, YY = np.meshgrid(np.linspace(extent[2],extent[3],G), np.linspace(extent[0],extent[1],G))
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = mlp.predict(pts)
    im = plt.imshow(output_pts.reshape((G,G)).T, vmin=0, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    plt.scatter(x[:,0], x[:,1], c=np.where(y==1, 'blue', 'red'))
    plt.show()
    
    # APPLY SCALAR
    
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)    
    
    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        x, 
        y, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0
    )
    
    #1A) [DONE]
    mlp = MLPClassifier(
        hidden_layer_sizes=(20,), #one hidden layer with 20 features
        activation='relu', #rectified linear
        learning_rate_init=1e-2, #learning rate
        max_iter=1000, # number of iterations
        early_stopping=True, #stop training if validation data gets worse
        random_state=0#random number seed for initialization
    ) 
    
    mlp.fit(x_train, y_train)
    
    output = mlp.predict(x_test)
    
    diff = np.array(y_test - output)
    
    rmse = np.sqrt(np.mean(np.square(diff)))
    
    print("1H RMSE (with scaler): ", rmse)
    
    extent = (-2, 2, -2, 2)
    G = 200
    XX, YY = np.meshgrid(np.linspace(extent[2],extent[3],G), np.linspace(extent[0],extent[1],G))
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = mlp.predict(pts)
    im = plt.imshow(output_pts.reshape((G,G)).T, vmin=0, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    plt.scatter(x[:,0], x[:,1], c=np.where(y==1, 'blue', 'red'))
    plt.show()
    
    discussion = \
    """
          The RMSE barely changes between the scaled and unscaled version. Given
          that the actual data scaling method is quite nebulous to me, and therefore
          could produce unexpected and undesirable results when new datasets are applied,
          I would not use the scaler here.
    """
    
    print(discussion)
    
    
    
    
    
    




