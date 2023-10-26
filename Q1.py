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
    
    #1A)
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
    
    print("1A RMSE: ", rmse)
    
    
    #1B)
    
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
        The network has learned to classify the data somewhat well.
        No blue points are misclassified, and the majority of red
        points are correctly classified, BUT there are a non-trivial amount
        of red points classified as blue.
    """
    
    #print(discussion)
    
    #1C)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(10,), #one hidden layer with 20 features
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
    
    print("1C RMSE: ", rmse)
    
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
        With less features, the classification is much worse. Red dots are
        classified only ~50% accurate, and blue dots are almost entirely 
        misclassified
    """
    
    #print(discussion)
    
    #1D)
    
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
    
    #1E)
    
    run_len = 10
    avgd_RMSEs = []
    
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
                hidden_layer_sizes=(feature_cnt,feature_cnt), #one hidden layer with 20 features
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
    plt.show()
    
    discussion = """
    The 2-layer network outperforms the 1 layer network for all numbers of features.
    This is generally expected, as having more features usually increases accuracy.
    However, the increase in perceptrons (double) does not result in 2x better
    accuracy. This suggests that the additional layer is not as useful as the first
    layer, and may hint at diminishing returns from additional layers.
    """
    
    #F)
    
    x, y = sklearn.datasets.make_circles(n_samples=100, shuffle=True, noise=0.1, random_state=0, factor=0.3)
    x[:,0]*=0.1
    
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
    
    print("1F RMSE: ", rmse)
    
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
        The accuracy is better than 1C but worse than 1A. This is to be expected,
        as there is less space between the two clusters.
    """

    #print(discussion)
    
    #G)
    
    x, y = sklearn.datasets.make_circles(n_samples=100, shuffle=True, noise=0.1, random_state=0, factor=0.3)
    x[:,0]*=0.1
    
    x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(
        x, 
        y, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0
    )
    
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)   
    
    
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
    
    
    #1H)
    
    x, y = sklearn.datasets.make_blobs(centers=[[-1, -1], [1, 1]], 
                                             cluster_std=[1, 1], 
                                             random_state=0, 
                                             n_samples=200, 
                                             n_features=2)

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
    
    print("1I RMSE (no scaler): ", rmse)
    
    extent = (-4, 4, -4, 4)
    G = 200
    XX, YY = np.meshgrid(np.linspace(extent[2],extent[3],G), np.linspace(extent[0],extent[1],G))
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = mlp.predict(pts)
    im = plt.imshow(output_pts.reshape((G,G)).T, vmin=0, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    plt.scatter(x[:,0], x[:,1], c=np.where(y==1, 'blue', 'red'))
    plt.show()
    
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test) 
    
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
    
    print("1I RMSE (with scaler): ", rmse)
    
    extent = (-4, 4, -4, 4)
    G = 200
    XX, YY = np.meshgrid(np.linspace(extent[2],extent[3],G), np.linspace(extent[0],extent[1],G))
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = mlp.predict(pts)
    im = plt.imshow(output_pts.reshape((G,G)).T, vmin=0, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    plt.scatter(x[:,0], x[:,1], c=np.where(y==1, 'blue', 'red'))
    plt.show()
    
    disussion = """
        Using the scaler does not increase the accuracy.
        Since this data overlaps in its original space, it will be much harder to seperate.
        However, since there are still two distinct blobs, it is possible that
        in a higher dimension, the data is more seperable. Therefore,
        adding more layers to the problem could increase the accuracy.
    """
    
    
    
    
    
    
    
    




