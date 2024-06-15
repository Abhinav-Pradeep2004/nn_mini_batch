import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

import pipeline as pl

z = [None]*config.NUM_LAYERS #weighted_sum
h = [None]*config.NUM_LAYERS


del_fl_by_del_z = [None]*config.NUM_LAYERS
del_hl_by_del_theta0 = [None]*config.NUM_LAYERS
del_hl_by_del_theta = [None]*config.NUM_LAYERS
del_L_by_del_h = [None]*config.NUM_LAYERS
del_L_by_del_theta0 = [None]*config.NUM_LAYERS
del_L_by_del_theta = [None]*config.NUM_LAYERS


def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs,current_layer_neurons_weights)



def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):

  if current_layer_neurons_activation_function == "linear":
    return current_layer_neurons_weighted_sums

  elif current_layer_neurons_activation_function == "sigmoid":
    return 1/(1 + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "tanh":
    return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums))/ \
            (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "relu":
    return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)
  


def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):

    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)

    elif current_layer_neurons_activation_function == "sigmoid":
        current_layer_neurons_outputs = 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
        return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)

    elif current_layer_neurons_activation_function == "tanh":
        return (2/(np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums)))**2

    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0)
    


def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
  return current_layer_neurons_outputs_dels



def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs,current_layer_neurons_outputs_dels):
  return np.matmul(previous_layer_neurons_outputs.T,current_layer_neurons_outputs_dels)



def run_training(tol,epsilon):
   
    epoch_counter = 0
    max_epoch = 10000
    
    mse = 1
    loss_per_epoch = list()
    loss_per_epoch.append(mse)

    training_data = load_dataset("train.csv")
    

    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:,0:2], training_data.iloc[:,2])
    X_train, Y_train = obj.transform(training_data.iloc[:,0:2], training_data.iloc[:,2])
    
    #initializing the parameters :random value of theta, theta0
    pl.initialize_parameters()
    #print("theta:",pl.theta)
    #print("theta0:",pl.theta0)
    

    while True:

      mse = 0

      for i in range(X_train.shape[0]):

        h[0] = X_train[i].reshape(1,X_train.shape[1])
        
        for l in range(1,config.NUM_LAYERS):
          
          z[l] = layer_neurons_weighted_sum(h[l-1], pl.theta0[l], pl.theta[l])
          
          h[l] = layer_neurons_output(z[l], config.f[l])
          
          del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l],z[l])
          
          del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
          
          del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l-1],del_fl_by_del_z[l])
          
        Y_train[i] = Y_train[i].reshape(Y_train[i].shape[0],1)     
            
        L = (1/2)*(Y_train[i][0] - h[config.NUM_LAYERS-1][0,0])**2

        mse = mse + L
        #print("mse:",mse)
        
        del_L_by_del_h[config.NUM_LAYERS-1] = (h[config.NUM_LAYERS-1] - Y_train[0])
        
        for l in range(config.NUM_LAYERS-2,0,-1):

          del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l+1], (del_fl_by_del_z[l+1] * pl.theta[l+1]).T)
        
        for l in range(1,config.NUM_LAYERS):

          del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
          del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]
          #print(del_hl_by_del_theta0)

          pl.theta0[l] = pl.theta0[l] - (epsilon * del_L_by_del_theta0[l])
          pl.theta[l] = pl.theta[l] - (epsilon * del_L_by_del_theta[l])
          

      mse = mse/X_train.shape[0]
      epoch_counter = epoch_counter + 1
      loss_per_epoch.append(mse)

      #print("Epoch # {}, Loss = {}".format(epoch_counter,mse))
      

      if (epoch_counter>=max_epoch):
         break
      
       
if __name__ == "__main__":
   run_training(10**(-4),10**(-5))
   save_model(pl.theta0,pl.theta)
   