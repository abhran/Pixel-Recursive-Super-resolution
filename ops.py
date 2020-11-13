import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
from tensorflow.keras import activations,layers
from tensorflow import keras



# class MaskedConvLayer(layers.Layer):
#     def __init__(self, mask_type, **kwargs):
#         super(MaskedConvLayer, self).__init__()
#         self.mask_type = mask_type
#         self.conv = layers.Conv2D(**kwargs)

#     def build(self, input_shape):
#         # Build the conv2d layer to initialize kernel variables
#         self.conv.build(input_shape)
#         # Use the initialized kernel to create the mask
#         kernel_shape = self.conv.kernel.get_shape()
#         self.mask = np.zeros(shape=kernel_shape)
#         self.mask[: kernel_shape[0] // 2, ...] = 1.0
#         if self.mask_type=="A":
#           self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
#         if self.mask_type == "B":
#           self.mask[kernel_shape[0] // 2, :(kernel_shape[1] // 2)+1, ...] = 1.0


#     def call(self, inputs):
#         self.conv.kernel.assign(self.conv.kernel * self.mask)
#         return self.conv(inputs)
    
#     def get_config(self):

#         config = super().get_config().copy()
#         config.update({
#             'mask_type': self.mask_type
#         })
#         return config








def MaskedConvLayer(x,mask_type,**kwargs):
   
    conv = keras.layers.Conv2D(**kwargs)

    input_shape=x.shape

    conv.build(input_shape)

    # Use the initialized kernel to create the mask
    
    kernel_shape = conv.kernel.get_shape()
    mask = np.zeros(shape=kernel_shape)
    mask[: kernel_shape[0] // 2, ...] = 1.0
    if mask_type=="A":
      mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
    if mask_type == "B":
      mask[kernel_shape[0] // 2, :(kernel_shape[1] // 2)+1, ...] = 1.0

    conv.kernel.assign(conv.kernel * mask)
    return conv(x)









# class ResidualBlock(keras.layers.Layer):
#     def __init__(self, filt,k_s,**kwargs):
#         super(ResidualBlock, self).__init__(**kwargs)
#         self.conv1 = keras.layers.Conv2D(filters=filt,kernel_size=k_s,**kwargs,padding='same')
#         self.bn1=keras.layers.BatchNormalization(center=True,scale=True,trainable=True)
#         self.relu1=keras.layers.ReLU()
#         self.conv2 = keras.layers.Conv2D(filters=filt,kernel_size=k_s,**kwargs ,padding='same' )
#         self.bn2=keras.layers.BatchNormalization()
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x= self.relu1(x)
#         x = self.conv2(x)
#         x= self.bn2(x)
#         return keras.layers.add([inputs, x])
#     def get_config(self):

#         config = super().get_config().copy()
#         config.update({
#           'filters':self.filt,
#           'kernel_size':self.k_s
#         })
#         return config
      

def  ResidualBlock(inputs,filters,kernel_size):    
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size ,padding='same')(inputs)
        x=keras.layers.BatchNormalization(center=True,scale=True,trainable=True)(x)
        x=keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size ,padding='same' )(x)
        x=keras.layers.BatchNormalization()(x)
        return keras.layers.add([inputs, x])






def gated_conv2d(inputs, state, kernel_shape):
  
  batch_size, height,  width,in_channel=inputs.get_shape().as_list()
  kernel_h, kernel_w = kernel_shape
  #state route
  # left =  MaskedConvLayer( mask_type='C', filters=2 * in_channel, kernel_size=kernel_shape, strides=[1, 1],padding='same')(state)
  left =  MaskedConvLayer( x=state,mask_type='C', filters=2 * in_channel, kernel_size=kernel_shape, strides=[1, 1],padding='same')
  left1 = left[:, :, :, 0:in_channel]
  left2 = left[:, :, :, in_channel:]
  left1=keras.layers.Activation(activations.tanh)(left1)
  left2 = keras.layers.Activation(activations.sigmoid)(left2)
  # left2 = tf.nn.sigmoid(left2)
  new_state = left1 * left2
  left2right = keras.layers.Conv2D( filters=2 * in_channel, kernel_size=1, strides=[1, 1],padding='same')(left)
  #input route
  # right = MaskedConvLayer( mask_type='B',filters= 2 * in_channel, kernel_size=kernel_shape, strides=[1, 1],padding='same')(inputs)
  right = MaskedConvLayer(x=inputs,mask_type='B',filters= 2 * in_channel, kernel_size=kernel_shape, strides=[1, 1],padding='same')
  right = right + left2right
  right1 = right[:, :, :, 0:in_channel]
  right2 = right[:, :, :, in_channel:]
  right1=keras.layers.Activation(activations.tanh)(right1)
  right2 = keras.layers.Activation(activations.sigmoid)(right2)
  up_right = right1 * right2
  up_right = MaskedConvLayer( x=up_right,mask_type='B', filters=in_channel, kernel_size=1, strides=[1, 1],padding='same')
  outputs = inputs + up_right

  return outputs, new_state


def batch_norm(x, train=True, scope=None):
  return tf.layers.batch_normalization(x, center=True, scale=True, trainable=True)



def deconv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], scope="deconv2d"):

  with tf.variable_scope(scope) as scope:
    return tf.layers.conv2d_transpose(inputs, num_outputs, kernel_shape, strides, \
          padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), \
          bias_initializer=tf.constant_initializer(0.0))