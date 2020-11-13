import tensorflow as tf 
from tensorflow import keras
import numpy as np
from ops import *


class Net(object):
  def __init__(self):
    """
    Args:[0, 255]
      hr_images: [batch_size, hr_height, hr_width, in_channels] float32
      lr_images: [batch_size, lr_height, lr_width, in_channels] float32
    """
  def prior_network(self, inputs,name):
      # x = MaskedConvLayer( mask_type="A", filters=64, kernel_size=7,strides=1, padding="same" )(inputs)
      x = MaskedConvLayer( x=inputs,mask_type="A", filters=64, kernel_size=7,strides=1, padding="same" )
      state=x 
      inputs=x
      for _ in range(20):
        inputs, state = gated_conv2d(inputs, state, [5, 5])
      # x = MaskedConvLayer(mask_type="B", filters=1024,kernel_size=1,strides=1, activation="relu", padding="same" )(x)
      # x = MaskedConvLayer(mask_type="B", filters=3*256,kernel_size=1,strides=1, padding="same" )(x)
      x = MaskedConvLayer(x=x,mask_type="B", filters=1024,kernel_size=1,strides=1, activation="relu", padding="same" )
      x = MaskedConvLayer(x=x,mask_type="B", filters=3*256,kernel_size=1,strides=1, padding="same" )
      prior_logits = tf.concat([x[:, :, :, 0::3], x[:, :, :, 1::3], x[:, :, :, 2::3]], 3 ,name=name)
      return prior_logits


  def conditioning_network(self, inputs,name):
      x=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1)(inputs)
      for i in range(2):
          for _ in range(6):
              # x = ResidualBlock(filt=32,k_s=3)(x)
              x = ResidualBlock(x,filters=32,kernel_size=3)
              x=relu1=keras.layers.ReLU()(x)
          x=keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal( mean=0.0, stddev=0.1, seed=None), bias_initializer=tf.constant_initializer(0.0),padding='SAME')(x)
      for _ in range(6):
          x = ResidualBlock(x,filters=32,kernel_size=3)
          # x = ResidualBlock(filt=32,k_s=3)(x)
          x=relu1=keras.layers.ReLU()(x)

      conditioning_logits = keras.layers.Conv2D( 3*256, kernel_size=1, strides=1,name=name)(x)
      return conditioning_logits

  def softmax_loss(labels,logits):
    logits = tf.reshape(logits, [-1,256 ])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels, logits)

          
      


