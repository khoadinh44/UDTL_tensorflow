import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
  return np.float(2.0 *(high - low)/(1.0 + np.exp(-alpha*iter_num/max_iter)) - (high-low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1
  
def AdversarialNet(in_feature, hidden_size):
  x = Dense(hidden_size, 
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
            activation='relu')(in_feature)
  x = Dropout(0.1)(x) 
  x = Dense(hidden_size, 
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
            activation='relu')(hidden_size)
  x = Dropout(0.1)(x)
  x = Dense(hidden_size, 
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
            activation='sigmoid')(1)
  return x
