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
    
    
    #G) [DONE]
    
    x, y = sklearn.datasets.make_circles(n_samples=100, shuffle=True, noise=0.1, random_state=0, factor=0.3)
    x[:,0]*=0.1
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
    
    print("1G RMSE: ", rmse)
    
    extent = (-2, 2, -2, 2)
    G = 200
    XX, YY = np.meshgrid(np.linspace(extent[2],extent[3],G), np.linspace(extent[0],extent[1],G))
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = mlp.predict(pts)
    im = plt.imshow(output_pts.reshape((G,G)).T, vmin=0, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    plt.scatter(x[:,0], x[:,1], c=np.where(y==1, 'blue', 'red'))
    plt.show()
    
    discussion = """
        The RMSE was decreased compared to 1F, but is now the exact same as 1A,
        hinting that the ideal scaling is that where the x and y spread are the same,
        ie, 2 cirlces.
    """

    #print(discussion)
    
    
    
    




