
# coding: utf-8

# ## Update
# -Add Batch Normalization layer after each conv
# -Add shuffle=True in model.fit() method for a better BN effect (so that we have different batch to normalize in each epoch during the training)
# -You can use crf method (https://www.kaggle.com/meaninglesslives/apply-crf) to improve the result
# ## Changelog
# - Changed uncov to uconv, but removed the dropout in the last layer
# - Corrected sanity check of predicted validation data (changed from ids_train to ids_valid)
# - Used correct mask (from original train_df) for threshold tuning (inserted y_valid_ori)
# - Added DICE loss functions

# # About
# Since I am new to learning from image segmentation and kaggle in general I want to share my noteook.
# I saw it is similar to others as it uses the U-net approach. I want to share it anyway because:
#
# - As said, the field is new to me so I am open to suggestions.
# - It visualizes some of the steps, e.g. scaling, to learn if the methods do what I expect which might be useful to others (I call them sanity checks).
# - Added stratification by the amount of salt contained in the image.
# - Added augmentation by flipping the images along the y axes (thanks to the forum for clarification).
# - Added dropout to the model which seems to improve performance.

# In[41]:


import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import sys
sys.path.append('../keras-deeplab-v3-plus')
from model import Deeplabv3
import glob

def downsample(img):
    return resize(img, (101, 101,3), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

ROOT = "/home/alexanderliao/data/Kaggle/datasets/pascal-voc-2012/VOC2012"
IMG_LIST = glob.glob(ROOT+'/SegmentationClass/*.png')
for i in range(len(IMG_LIST)):
    IMG_LIST[i]=IMG_LIST[i][83:len(IMG_LIST[i])-4]

from PIL import Image

def make_square(im, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, ( int((size-x)/2), int((size-y))))
    return new_im
IMG_LIST[1]

tvmonitor = np.array([0,64,-128])
ambiguous=np.array([-32,-32,-64])
train=np.array([-128,-64,0])
aeroplane=np.array([-128,0,0])
boat=np.array([  0   ,0,-128])
chair=np.array([-64 ,0 ,0])
dog=np.array([  64  ,0,-128])
bird=np.array([-128,-128,0])
diningtable=np.array([-64,-128 ,0])
bottle=np.array([-128   ,0,-128])
bicycle=np.array([   0,-128,0])
person=np.array([ -64 -128 -128])
motorbike=np.array([  64,-128,-128])
sheep=np.array([-128,64 ,0])
horse=np.array([ -64,0,-128])
cat=np.array([64,0 ,0])
bus=np.array([   0,-128,-128])
cow = np.array([  64,-128,0])
pottedplant=np.array([ 0,64,0])
sofa=np.array([  0,-64,  0])
background=np.array([0,0,0])
car=np.array([-128,-128,-128])

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#a=np.array(downsample(np.array(make_square(Image.open(ROOT+"/SegmentationClass/{}.png".format("2007_003143"))),dtype='int8')),dtype='int8')
#imgplot = plt.imshow(a)
#k=0
#for i in range(101):
#    print(a[70][i])



"""
for i in range(500):
    if not(  ( a[i][300]==np.array([0,0,0]) ).any() ):
        print(a[i][300])
        print(i)
        break
"""


# # Read images and masks
# Load the images and masks into the DataFrame and divide the pixel values by 255.

# In[6]:


train_images = [downsample(np.array(make_square(Image.open(ROOT+"/JPEGImages/{}.jpg".format(idx))))) / 255 for idx in tqdm_notebook(IMG_LIST)]
#train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


# In[ ]:


train_masks = [np.array(downsample(np.array(make_square(Image.open(ROOT+"/SegmentationClass/{}.png".format(idx))),dtype='int8')),dtype='int8') for idx in tqdm_notebook(IMG_LIST)]

#train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]



from tqdm import tqdm
int_masks = []
for i in tqdm(range(len(train_masks))):
    image=train_masks[i]
    new_im=np.zeros((101,101,1))
    for i in range(101):
        for j in range(101):
            if (image[i,j,:] == aeroplane).all():
                new_im[i,j,:]=1
            elif (image[i,j,:] == bicycle).all():
                new_im[i,j,:]=2
            elif (image[i,j,:] == bird).all():
                new_im[i,j,:]=3
            elif (image[i,j,:] == boat).all():
                new_im[i,j,:]=4
            elif (image[i,j,:] == bottle).all():
                new_im[i,j,:]=5
            elif (image[i,j,:] == bus).all():
                new_im[i,j,:]=6
            elif (image[i,j,:] == car).all():
                new_im[i,j,:]=7
            elif (image[i,j,:] == cat).all():
                new_im[i,j,:]=8
            elif (image[i,j,:] == chair).all():
                new_im[i,j,:]=9
            elif (image[i,j,:] == cow).all():
                new_im[i,j,:]=10
            elif (image[i,j,:] == diningtable).all():
                new_im[i,j,:]=11
            elif (image[i,j,:] == dog).all():
                new_im[i,j,:]=12
            elif (image[i,j,:] == horse).all():
                new_im[i,j,:]=13
            elif (image[i,j,:] == motorbike).all():
                new_im[i,j,:]=14
            elif (image[i,j,:] == person).all():
                new_im[i,j,:]=15
            elif (image[i,j,:] == pottedplant).all():
                new_im[i,j,:]=16
            elif (image[i,j,:] == sheep).all():
                new_im[i,j,:]=17
            elif (image[i,j,:] == sofa).all():
                new_im[i,j,:]=18
            elif (image[i,j,:] == train).all():
                new_im[i,j,:]=19
            elif (image[i,j,:] == tvmonitor).all():
                new_im[i,j,:]=20
            elif (image[i,j,:] == ambiguous).all():
                new_im[i,j,:]=21
    int_masks.append(new_im)


import pickle
pickle.dump( int_masks, open( "PASCAL_VOC2012_int_mask.p", "wb" ) )
int_masks = pickle.load( open( "PASCAL_VOC2012_int_mask.p", "rb" ) )
#pickle.dump( int_masks, open( "PASCAL_VOC2012_int_mask.p", "wb" ) )


"""
from keras.utils.np_utils import to_categorical
one_hot_masks = to_categorical(int_masks, num_classes=22)
del int_masks
"""

x_train, x_valid, y_train, y_valid = train_test_split(
   np.array(train_images).reshape(-1, 101, 101, 3),
   np.array(int_masks).reshape(-1, 101, 101, 1),
   test_size=0.2, random_state=1337)

model =  Deeplabv3(input_shape=(101,101,3),backbone="mobilenetv2", classes=22)
#model =  Deeplabv3(input_shape=(101,101,3),backbone="xception", classes=22)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy","sparse_categorical_crossentropy"])


# In[33]:


print(model.summary())



x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)



print(x_train.shape)
print(np.repeat(x_train,3,3).shape)


early_stopping = EarlyStopping(patience=50, verbose=1)
model_checkpoint = ModelCheckpoint("./deeplabv3.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=35, min_lr=0.00001, verbose=1)

epochs = 500
batch_size = 32

with tf.device ("/gpu:0"):
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[reduce_lr, model_checkpoint],shuffle=True)
