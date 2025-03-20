# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:15:46 2025

@author: simon

A scrip to train a model to determine if a cell is infected by CPV using the DNA/DAPI channel
Assume that the images are prepared and in one folder:
    - Image centered around the nuclei
    - One nuclei by image
    - Image resized to 100x100 pixels
    - Image in a 8-bits PNG format
    - NO Zstack or multichannel image
    - Image shape should be (100,100)
    - Images are normalized (0-255)
Training data are available in the GitHub repo: https://github.com/leclercsimon74/Infection_Prediction_CPV
Require 3 folders:
    - Infected cells
    - Noninfected cells
    - Out for focus cell/signal

It will output in the model folder:
    - multiple h5 files, the last should be the best.

To predict images infection using this model: Prediction of Infection CPV

Possible improvements:
    - add a gaussian filter in the augment
    - add noise (white and black) in the augment
    - coherence in image format between the training (PNG) and the prediction (TIFF)
     
"""
#MAIN parameters

NI_folder = r"data\Training\Check_NI"
INF_folder = r"data\Training\Checked_infected"
OUT_folder = r"data\Training\Check_Out"
epoch = 300

#global imports
import os
import numpy as np
import tifffile
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import re
from tensorflow.keras.callbacks import CSVLogger

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
#Machine learning part
class MyCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, freq=5, directory=r"data\Model"):
        """Save the weight of the model at specified interval"""
        super().__init__()
        self.freq = freq
        self.directory = directory

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq > 0 and epoch % self.freq == 0:
            filepath = self.directory +os.sep+"model_weight-"+str(epoch)
            self.model.save_weights(filepath)

    def on_train_end(self, logs=None):
        self.model.save_weights(self.directory+os.sep+"model_weight-last")


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

def augment_patches(patches):
    #logistic function that augment the data by rotation and flip (*8)
    augmented = np.concatenate((patches,
                                np.rot90(patches, k=1, axes=(1, 2)),
                                np.rot90(patches, k=2, axes=(1, 2)),
                                np.rot90(patches, k=3, axes=(1, 2))))

    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented

def found_img(path, extension='.tiff'):
    #logistic function to found all tif/extension file
    imgs = [f for f in os.listdir(path) if f.endswith(extension)]
    imgs_path = [path+os.sep+f for f in imgs]
    return imgs_path

def get_data(path, name):
    #get the images, augment them and adjust their dimension so to fit for ML
    img_paths = found_img(path, extension='png')
    imgs = []
    names = []
    for path in img_paths:
        # img = img = tifffile.imread(path)
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)
        img /= 255
        img = img[..., np.newaxis]
        img_a = augment_patches(img[np.newaxis,...])
        for img in img_a:
            names.append(name)
            imgs.append(img[..., np.newaxis])
    
    return np.array(imgs), names
  
def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#load and prepare the data
NI_data, namesNI = get_data(NI_folder, 'NI')
infected_data, namesInf = get_data(INF_folder, 'Infected')
out_data, namesOut = get_data(OUT_folder, 'Out')


data = np.append(NI_data, infected_data, axis=0)
data = np.append(data, out_data, axis=0)
class_name = namesNI+namesInf+namesOut
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

#split the data between train and validation
#done manually (can be done in tf) to keep some control and run some test on them
val = np.random.choice(np.arange(len(target_val)),
                       int(len(target_val)*0.15), #15% of the data used for validation
                       replace=False)
tokeep = np.zeros(len(target_val))
tokeep[val] = 1
tokeep = np.invert(np.array(tokeep, dtype='bool'))
#validation data
val_data = data[val]
val_target = np.array(target_val)[val]
val_target = tf.keras.utils.to_categorical(val_target)
#train data
train_data = data[tokeep]
train_target = np.array(target_val)[tokeep]
train_target = tf.keras.utils.to_categorical(train_target)

print('Data prepared')
print(target_dict.keys())

#make the untrained/naive model
cnn = make_model()

#train the machine learning
log = CSVLogger(r"data\Model\log.log", append=True) #make the logger saving the learning progress. Delete if training from 0
autosave = MyCheckpoint(freq=20) #autosave the weight 
rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.2,
                                              patience=5, mode='min') #reduce the learning rate if plateauing
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1,
                                      restore_best_weights=True) #stop early if no more improvement of the val_loss
history = cnn.fit(train_data, train_target,
                  validation_data=(val_data, val_target),
                  epochs=epoch, #Got stable result after 100 epochs for the val_accuracy
                  callbacks=[log, autosave, rlronp, es])

#evaluate on validation data
scores = cnn.evaluate(val_data, val_target, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#save the full model based on its score
cnn.save(r"data\Model\model1.1_"+str(round(scores[1]*100, 2))+".h5")

#show and save the history
#WARNING. If log has not been deleted between two trainings, the graph will show both!
df = pd.read_csv('data\Model\log.log')
df = df.drop(columns=['epoch'])

#accuracy plot
df.plot(y=['accuracy', 'val_accuracy'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.savefig('accuracy_history.png')
plt.show()

#loss plot
df.plot(y=['loss', 'val_loss'], logy=True)
plt.ylabel('Loss (categorical_crossentropy)')
plt.xlabel('Epoch')
plt.savefig('loss_history.png')
plt.show()

#assess if the machine learning recognize more easy one class compare to the other
inf = val_data[np.where(val_target[:,0]==1)]
ni = val_data[np.where(val_target[:,1]==1)]
out = val_data[np.where(val_target[:,2]==1)]

#individual evaluation of infected and non-infected
scores = cnn.evaluate(inf, val_target[np.where(val_target[:,0]==1)], verbose=0)
print("Accuracy infection: %.2f%%" % (scores[1]*100))
scores = cnn.evaluate(ni, val_target[np.where(val_target[:,1]==1)], verbose=0)
print("Accuracy non-infected: %.2f%%" % (scores[1]*100))
scores = cnn.evaluate(out, val_target[np.where(val_target[:,2]==1)], verbose=0)
print("Accuracy out: %.2f%%" % (scores[1]*100))

#simple prediction figure on validation data
#left are noninfected cell, with their noninfected score
#right are infected cell, with their infected score

#add some randomness
np.random.shuffle(inf)
np.random.shuffle(ni)
np.random.shuffle(out)
data = [ni, inf, out]

nrows = 5 #number of row to show

fig, ax = plt.subplots(figsize=(9, nrows*3) , nrows=nrows, ncols=3)

i=0
im = 0
for j, row in enumerate(ax):
    for k, col in enumerate(row):
        col.imshow(np.squeeze(data[k][im]))
        col.set_axis_off()
        #format data in shape (1,100,100,1)
        p = cnn.predict(np.squeeze(data[k][im])[np.newaxis,...,np.newaxis])
        if i == 0:
            name = 'NI, score at: '
            p = p[0][1]
        elif i == 1:
            name = 'Infected, score at: '
            p = p[0][0]
        else:
            name = 'Out, score at: '
            p = p[0][2]
        col.set_title(name+str(round(p, 4)))
        if i <= 1:
            i += 1
        else:
            i = 0
    im += 1

plt.tight_layout()
plt.show()