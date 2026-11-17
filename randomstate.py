from sklearn.datasets import make_regression
import numpy as np
seed = 3
for i in range(seed):
    X,y  = make_regression(
        n_samples=50, 
        n_features=3, 
        n_targets=1, 
        random_state=np.random.default_rng(i)
        )
    print("input")
    print (X)