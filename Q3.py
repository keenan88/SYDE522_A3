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
    
    """
    
    Invesigation topic: Predicting goal-scoring per game in NHL hockey (likelihood of scoring 0, 1, 2, or 3 goals)
    
    Dataset: I have been building up a dataset of NHL player, game, and schedule data from:
    https://gitlab.com/dword4/nhlapi/-/blob/master/swagger/openapi.yaml?ref_type=heads.
    
    You can see the dataset I have built up at: https://github.com/keenan88/FH_Tools_V3/tree/main/Database/NHL
    
    The dataset includes (and can be reformatted to include) things like:
    player position, team, ice time, schedule density, recent performance (I'm very
    interested to see if the hot hand fallacy is indeed a fallacy), goals per game,
    opponent goals allowed per game, goal scoring as the season progresses, etc.
    
    I suspect that player's goals per game (gpg) and the number of goals their opponents
    let in (gapg) on average will be the best indicators, as general indicators of player's 
    goal scoring ability and opponents defense ability. Accordingly, I will take a 
    sample of the current NHL 2023-2024 season, take all games played by all players,
    and for each player-game,  calculate the player's past goals/game and opponents past 
    allowed goals/game.
    
    With that dataset, I will try a regression Y = XW, where Y is a column vector of goals
    scored in each player-game, X's rows are the gpg and opp gapg and columns are each player
    game, and W is a 2x1 weight matrix. I might try some various bases, like polynomial
    or guassian, depending on how accurate a first lienar attempt is.
    
    I am also very interested to see how this data look against an SVM, as I am
    expecting each classification group to have signifcant overlap in the feature space.
    Experimenting with gamma and C will be fun, to try and figure out the optimal tradeoff
    between testing data being less accurate but generalizing better.
    
    Finally, I will try classifying with an MLP classifier, with various numbers of layers
    and layer sizes. I am a little more skeptical of this approach, as it is much more 
    convoluted as two what classification is actually happening.
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    
    
    

    
    
    
    
    
    




