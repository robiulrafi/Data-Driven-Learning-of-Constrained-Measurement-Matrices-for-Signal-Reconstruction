#!/usr/bin/env python
# coding: utf-8
## Importing the required Libraries

import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense,Reshape,Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import keras.initializers
from keras.optimizers import Adam
from Model_recon import *
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import scipy.io as sio 
import scipy.linalg as linalg
from keras import backend as K
import os
from keras.layers import Lambda
batch_size =512
nb_epoch = 100
img_rows, img_cols = 32, 32
from keras.callbacks import ModelCheckpoint




## Initializing the MM for a given measurement
M=128
PHI=np.random.randn(M,32*32)
PHI=linalg.orth(PHI.T).T
PHI_ex=PHI.T

## Number of DNN Stages
n=2
X_in=Input(shape=(32,32,1,))
layers,_,_=bulid_reconstruction(X_in,PHI_ex,M,n)


## Loss related to reconstruction

cost1=tf.losses.mean_squared_error(labels=X_in,predictions=layers[0])
cost2=tf.losses.mean_squared_error(labels=X_in,predictions=layers[1])
cost3=tf.losses.mean_squared_error(labels=X_in,predictions=layers[2])
cost=0.33*(cost1+cost2+cost3)


## Given keras Model
model = Model(inputs=[X_in], outputs=[layers[-1]])




## Constraining the MM
d=model.trainable_weights[0]
## Loss term after adding the constraint
e=(1+d)*(1+d)*(1-d)*(1-d)


## Loss function
def penalized_loss(noise,d,alpha):
    TL=noise+alpha*K.sum(K.sum(d,axis=0),axis=0)
    def loss(y_true, y_pred):
        return TL
    return loss


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
def psnr(noise,X_in):
    mse = tf.losses.mean_squared_error(labels=X_in,predictions=noise)
    def loss(y_true, y_pred):
        
        psnr_e=20 * log10(255.0 / tf.math.sqrt(mse))
        return psnr_e

    return loss
model.compile(loss=[penalized_loss(cost,e,2.0)],optimizer=Adam(lr=1e-4),metrics=[psnr(layers[-1],X_in)])


## Loading the training and validation data
Training_Image=sio.loadmat('C:\\RAFI_SHARED\\CS_NET\\GAN\\Training_Image_Label.mat')['X_out']
Testing_Image=sio.loadmat('C:\\RAFI_SHARED\\CS_NET\\GAN\\validation_Image_Label.mat')['X_out']
Training_Image=np.reshape(Training_Image,(Training_Image.shape[0],32,32,1))
Testing_Image=np.reshape(Testing_Image,(Testing_Image.shape[0],32,32,1))
X_train=Training_Image
X_test=Testing_Image
idx=np.random.randint(50000,size=50000)
P=idx[:40000]
Q=idx[40000:]
TI=Training_Image[P,:]
VI=Training_Image[Q,:]
X_train=TI
X_val=VI

## Training
model.fit([X_train], [X_train],epochs=100,batch_size=32,
          validation_data=([X_val],[X_val] ))


## Inference Phase
## Rouding of the MM
LO=model.get_weights()
QW=np.round(LO[0])


## Replacing the weights 
model.layers[2].set_weights([QW,LO[1]])   
layer_name = 'activation_6'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict([X_test])

## PSNR calculation
imgs, row, col,_= intermediate_output.shape
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)),math.sqrt(mse)

Y_dev=X_test
Y_recon=intermediate_output
J=np.zeros((imgs,1))
T=np.zeros((imgs,1))
for i in range(imgs):
        J[i],T[i]=(psnr((Y_recon[i]),(Y_dev[i])))
## PSNR Values
print(np.mean(J))
print(np.mean(T))







