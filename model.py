### Importing packages.

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Flatten
from keras.layers import Dropout, Lambda, ELU, Cropping2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import initializations
from keras.models import model_from_json
from keras import backend as K
import json

from helper import generate_next_batch
import matplotlib.pyplot as plt



# NVIDIA model
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

def get_model():

    input_shape = (160, 320, 3)
    start = 0.001
    stop = 0.001
    nb_epoch = 10

    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape=input_shape,output_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    # learning_rates = np.linspace(start, stop, nb_epoch)
    # change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
    # sgd = SGD(lr=start, momentum=0.9, nesterov=True)
    adam = Adam(lr=start, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam)
    return model

# Get the model
model = get_model()

# Train the model
batch_size = 256
number_of_samples_per_epoch = 20480

def split(csv, val_split):
	shuffled = csv.iloc[np.random.permutation(len(csv))]
	validation_samples = int(len(csv) * val_split)
	return (shuffled[validation_samples:],
				shuffled[:validation_samples])


## Import data
# Added header row manually to CSV.
# driving_csv = pd.read_csv("data/course1_driving_log.csv")
driving_csv = pd.read_csv("data/course3_driving_log.csv")

# Examine data
print("Number of datapoints: %d" % len(driving_csv))

driving_csv.head()

train_data, val_data = split(driving_csv, 0.2)
number_of_validation = len(val_data)

train_generate = generate_next_batch(train_data)
validation_generate = generate_next_batch(val_data)

history_object = model.fit_generator(train_generate,
                  samples_per_epoch = 20480,
                  nb_epoch = 10,
                  validation_data = validation_generate,
                  nb_val_samples = number_of_validation,
                  verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

print('Save the model')

#model.save_weights('./model.h5')
model.save('model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)

print('Done')

K.clear_session()
