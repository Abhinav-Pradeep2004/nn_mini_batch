import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

import pipeline as pl
import train_pipeline as tl

def predict(X,t0,t,a):
    o = []
    for i in range(X.shape[0]):

        tl.h[0] = X[i].reshape(1,X.shape[1])
        for l in range(1, config.NUM_LAYERS):
            tl.z[l] = tl.layer_neurons_weighted_sum(tl.h[l-1], t0[l], t[l])
            tl.h[l] = tl.layer_neurons_output(tl.z[l], a[l])
        
        o.append(tl.h[config.NUM_LAYERS - 1][0][0])

    return o
if __name__ == "__main__":
    X_test = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])  


    params = load_model("two_input_xor_nn.pkl") 
    theta0 = params["params"]["biases"]
    theta = params["params"]["weights"]
    activation = params["activations"]
    output = predict(X_test,theta0,theta,activation)
    pred = []
    percent = 0
    for i in output:
        if(i<0.5):
            pred.append(0)
        else:
            pred.append(1)
    for p in range(len(pred)):
        print(f"Input: {X_test[p]}, Predicted Output: {pred[p]}")
        percent += pred[p]
    
    if percent == 3:
        print("Accuracy:75%")
        
        
    
    
    
        
        
            
        
        