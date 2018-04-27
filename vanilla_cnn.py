# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
np.random.seed(111) # for 'reproducibilty' 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# My stuff
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# More
import io
import glob
import cv2
from pathlib import Path
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from skimage.io import imread
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
color = sns.color_palette()
#%matplotlib inline


# Define some paths first
input_dir = Path('../input')
images_dir = input_dir / 'train_color'
labels_dir = input_dir / 'train_label'



train_images = sorted(os.listdir(images_dir))
train_labels = sorted(os.listdir(labels_dir))

print("Number of images and labels in the training data: {}  and {} respectively".format(len(train_images), len(train_labels)))


# Define the label mappings 
labelmap = {0:'others', 
            1:'rover', 
            17:'sky', 
            33:'car', 
            34:'motorbicycle', 
            35:'bicycle', 
            36:'person', 
            37:'rider', 
            38:'truck', 
            39:'bus', 
            40:'tricycle', 
            49:'road', 
            50:'siderwalk', 
            65:'traffic_cone', 
            66:'road_pile', 
            67:'fence', 
            81:'traffic_light', 
            82:'pole', 
            83:'traffic_sign', 
            84:'wall', 
            85:'dustbin', 
            86:'billboard', 
            97:'building', 
            98:'bridge', 
            99:'tunnel', 
            100:'overpass', 
            113:'vegatation', 
            161:'car_groups', 
            162:'motorbicycle_group', 
            163:'bicycle_group', 
            164:'person_group', 
            165:'rider_group', 
            166:'truck_group', 
            167:'bus_group', 
            168:'tricycle_group'}


# Create an empty dataframe
data_df = pd.DataFrame()
df_list = []



# Iterate over data. I have just shown it for 500 images just to save time 
for idx in range(50):
    print("Working on Image #" + str(idx))
    # Get the image name and corresponding label
    img_name = train_images[idx]
    label_name = train_labels[idx]
    label = imread(labels_dir / train_labels[idx])
    pixel_classes = np.unique(label//1000)
    classes, instance_count = np.unique(pixel_classes, return_counts=True) # Courtesy:https://www.kaggle.com/jpmiller/cvpr-eda
    data_dict = dict(zip(classes, instance_count))
    df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    df.rename(columns=labelmap, inplace=True)
    df['img'] = img_name
    df['label'] = label_name
    
    # Concate to the final dataframe
    #data_df = pd.concat([data_df, df], copy=False)
    # append to the list of intermediate df list
    df_list.append(df)
    
data_df = pd.concat(df_list, axis=0)
del df_list

# Fill the NaN with zero
data_df = data_df.fillna(0)

# Rearrange the columns
cols = data_df.columns.tolist()
cols = [x for x in cols if x not in ['img', 'label']]
cols = ['img', 'label'] + cols
data_df = data_df[cols]

# Display the results
data_df = data_df.reset_index(drop=True)
data_df.head(10)    

print("Taking a subset of images")
sample_images = (data_df['img'][10:15]).reset_index(drop=True)
sample_labels = (data_df['label'][10:15]).reset_index(drop=True)


print("Setting up Plots")
f, ax = plt.subplots(5,3, figsize=(20,20))

data = []
labels  = []

for i in range(5):
    print("Viewing Image #" + str(i))
    img = imread(images_dir / sample_images[i])
    label = imread(labels_dir / sample_labels[i]) // 1000
    label[label!=0] = 255
    blended_image = Image.blend(Image.fromarray(img), Image.fromarray(label).convert('RGB'), alpha = 0.8)
    
    data.append(img)
    labels.append(img)
    
    ax[i, 0].imshow(img, aspect='auto')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(label, aspect='auto')
    ax[i, 1].axis('off')
    ax[i, 2].imshow(blended_image, aspect='auto')
    ax[i, 2].axis('off')

plt.show()


X_train = np.asarray(data)
Y_train = np.asarray(labels)
# Setting Up Neural Networks
model = Sequential()


# Layer 1
model.add(Convolution2D(32,3,3, activation='relu', input_shape=(2710,3384,3)))

# Layer 2
model.add(Convolution2D(32,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # Prevents over fitting: https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning

# Layer 3
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compling and Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)