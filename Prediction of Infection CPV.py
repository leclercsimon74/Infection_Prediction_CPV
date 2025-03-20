# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:18:00 2025

@author: simon

A prediction model to determine if a cell is infected by CPV using the DNA/DAPI channel
Assume that the images are prepared and in one folder:
    - Image centered around the nuclei
    - One nuclei by image
    - Image resized to 100x100 pixels
    - Image in a 8-bits TIFF format
    - NO Zstack or multichannel image
    - Image shape should be (100,100)
    - Better if the images are normalized (0-255)
The model is present in the GitHub repo: https://github.com/leclercsimon74/Infection_Prediction_CPV

It is also possible to train you own model: CNN Training for Infection CPV

It will output in the SAME folder to predict (so with the images):
    - One CSV file (Result.csv) with different probabilities:
        - Infection score, 0-1
        - Noninfected score, 0-1
        - Out score (for out of focus/blurry image), 0-1
        - And the name of the image
    - A visualization image, that will display the {max_img_nb} first images
        with their name and infection score if visualization is set to True

"""

#MAIN PARAMETERS
to_predict_path = r"data\Images"
model_name = r"data\Model\model1.1_93.74.h5"

visualization = True #bool
max_img_nb = 25 #int

#global imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tifffile
from skimage.transform import resize

#Import tensorflow and check for GPU access
import tensorflow as tf
print('TensorFlow version:'+tf.__version__)
if tf.test.gpu_device_name()=='':
  print('You do not have GPU access.')
  print('Did you change your runtime ?')
  print('If the runtime setting is correct then Google did not allocate a GPU for your session')
  print('Expect slow performance. To access GPU try reconnecting later')

else:
  print('You have GPU access')

#function used
def make_model():
    """simple model for discrimination. Three conv2D with different filters size
    to detect small features of the image.
    The image input sould be 100 by 100 pixel in one color (1 channel).
    Return the model"""
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Input(shape=(100,100,1)))
    cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.Dense(24, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    cnn.add(tf.keras.layers.Dropout(0.3))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dense(3, activation='softmax'))
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                metrics=['accuracy'])

    return cnn

def found_img(path:str, extension='.tiff'):
    #logistic function to found all tif/extension file
    imgs = [f for f in os.listdir(path) if f.endswith(extension)]
    imgs_path = [path+os.sep+f for f in imgs]
    return imgs_path


def sorted_nicely(l:list):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_data(path:str):
    #small function to grab the images and prepare them in a similar way to the training dataset
    img_paths = found_img(path)
    img_paths = sorted_nicely(img_paths)
    imgs = []
    names = []
    for p in img_paths:
        img = tifffile.imread(p)
        img = np.array(img, dtype=np.float32)
        img = resize(img, (100,100)) #to be sure!
        img /= 255 #normalize 0-1
        imgs.append(img)
        names.append(p)

    return imgs, names

def get_model(model_name:str=''):
    #load an existing model
    model_name_list = found_img(os.getcwd(), extension='.h5')
    weight = [x for x in os.listdir(os.getcwd()) if 'model_weight-' in x]
    
    if model_name != '':
        cnn = tf.keras.models.load_model(model_name)
        print('Load the named model: '+model_name)
    elif len(model_name_list) != 0: #from a clean saved model
        model_name_list = sorted_nicely(model_name_list)
        model_name = model_name_list[-1] #highest score
        cnn = tf.keras.models.load_model(model_name)
        print('Load the full model: '+model_name)
    elif len(weight) != 0: #from a security saved, #WARNING, assuming the SAME model!!
        last = [x for x in weight if 'model_weight-last' in x]
        if len(last) != 0:
          weight = sorted_nicely(weight)
          weight = weight[-1]
        else:
          weight = last
        cnn = make_model()
        cnn.load_weights(weight)
    else:
        raise Exception('No model found in the folder')
    
    return cnn


img_list, img_name = get_data(to_predict_path)

img_arr = np.array(img_list)
img_arr = img_arr[..., np.newaxis] #need a channel dimension

cnn = get_model(model_name)
prediction = cnn.predict(img_arr) #prediction here

# Data organization and saving
df = pd.DataFrame(data=img_name, columns=['Name'])
df['Infected score'] = prediction[:,0]
df['Noninfected score'] = prediction[:,1]
df['Out score'] = prediction[:,2]
df.to_csv(to_predict_path+os.sep+'Results.csv')
#%%
#figure
if visualization:
    if len(df)>max_img_nb:
        new_df = df[:max_img_nb]
    else:
        new_df = new_df.copy()
    nrows = len(new_df)//5 #5 images by rows
    fig, ax = plt.subplots(figsize=(15, nrows*3) , nrows=nrows, ncols=5)
    
    im = 0
    for j, row in enumerate(ax):
        for k, col in enumerate(row):
            if im < len(new_df):
                col.imshow(img_list[im])
                score = 'Infected score at: '
                score = score+str(round(prediction[im][0], 4))
                col.set_title(os.path.basename(img_name[im].replace('.tif', ''))[:25]+'\n'+score)
    
            col.set_axis_off()
            im += 1
    
    plt.tight_layout()
    plt.savefig(to_predict_path+os.sep+'Prediction_results.png', dpi=150)
    plt.show()

