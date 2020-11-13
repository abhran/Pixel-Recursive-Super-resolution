import tensorflow as tf 
import numpy as np 
from keras.models import load_model
from keras.models import model_from_json
from PIL import Image
import time
import os
from tqdm import tqdm
from os import path
import time
from tensorflow import keras
import matplotlib.pyplot as plt
from ops import *
from data import *
from net import *
from utils import *
import os
import time
import pandas as pd
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
import keras.objectives
from keras.callbacks import CSVLogger
net=Net()



tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='tensorboard_rerun',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

model_no=5000
batch_size=32

csv_logger = CSVLogger(f'history/loss_log.csv', append=True, separator=',')



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

json_file = open(f'model{model_no}/model{model_no}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"model{model_no}/model{model_no}.h5")
print("Loaded model from disk")
pixel_cnn=loaded_model




data=DataSet('datafolder',8,32,3)
hr_images,lr_images=data.hr,data.lr


def softmax_loss( labels,logits):
    logits = tf.reshape(logits, [-1,256 ])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
           labels, logits)


losses = {
	
	"cond": softmax_loss,
    "combined":softmax_loss
}


lossWeights = { "cond": 1.0,"combined":1.0}
opt=keras.optimizers.RMSprop(lr=4e-4,decay=0.95,momentum=0.9, epsilon=1e-8, name="RMSprop")


# pixel_cnn = keras.Model(inputs=[cond_inputs,prior_inputs],outputs=[x,x_,combined])
print("[INFO] compiling model...")
print(pixel_cnn.summary())

pixel_cnn.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)#, metrics=["accuracy"])


start_time = time.time()
t=0
hr_image = hr_images /127.5-1.0
lr_image = lr_images /127.5-1.0




def lr_scheduler(epoch, lr):
    if epoch <= 500:
        return 4e-5
    elif epoch>500 and epoch<=5000:
        return lr * 0.8**(2000/500000)
    elif epoch>5000 and epoch<=10000:
        return lr * 0.6**(5000/500000)
    elif epoch>10000 and epoch<=20000:
        return lr * 0.6**(10000/500000)
    else:
        return lr * 0.6**(20000/500000)

callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)




cp1= ModelCheckpoint(filepath="model/save_best.h5", monitor='loss',save_best_only=True, mode='min',verbose=1,save_weights_only=True)
cp2= ModelCheckpoint(filepath='model/save_all.h5', monitor='loss',save_best_only=False ,verbose=1,save_weights_only=True)
# callbacks_list = [callback,cp1,cp2,csv_logger]
callbacks_list = [callback,cp1,cp2,tensorboard_cb,csv_logger]

print(round(pixel_cnn.optimizer.lr.numpy(), 5))
pixel_cnn.fit(x={"priorin": hr_image, "condin": lr_image}, y={"cond":hr_images ,"combined":hr_images} , epochs=25000,batch_size=batch_size,callbacks=[callbacks_list])
print(round(pixel_cnn.optimizer.lr.numpy(), 5))



model_json = pixel_cnn.to_json()
with open(f"last_model/model.json", "w") as json_file:
    json_file.write(model_json)
pixel_cnn.save_weights(f"last_model/model.h5") 
       


end_time = time.time()
t=t-start_time+end_time

print("time taken in training is : ",t," sec")