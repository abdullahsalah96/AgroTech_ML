# from scipy.misc import imread
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.models import model_from_json, load_model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras import metrics
import numpy as np
from glob import glob
from skimage import color
from tqdm import tqdm
import cv2
import os
from helper import HelperCallbacks, HelperImageGenerator, ImagesLoader, Annotations


class BottleneckModel():
    def __init__(self, pre_model, pooling, weights, input_shape, is_trainable):
        self.pre_model = pre_model
        self.model = Sequential()
        self.pooling = pooling
        self.weights = weights
        self.input_shape = input_shape
        self.is_trainable = is_trainable

    def load_bottleneck_model(self):
        """
        A function that returns the pre-trained model
        """
        self.model.add(self.pre_model(include_top = False, pooling = self.pooling, weights = self.weights, input_shape = self.input_shape))
        if(self.is_trainable):
            self.model.layers[0].trainable = True
        else:
            self.model.layers[0].trainable = False
        return (self.model)


#importing the training and testing images
imageLoader = ImagesLoader()
train_images, tr_labels = imageLoader.load_images(r"/home/abdullahsalah96/IBM/Dataset/Training/", 3)
test_images, ts_labels = imageLoader.load_images(r"/home/abdullahsalah96/IBM/Dataset/Testing/", 3)

#converting the training and testing images to 4d tensors to be fed to the convolutional layers
train_tensors = imageLoader.paths_to_tensor(train_images, (32,32)).astype('float32')
test_tensors = imageLoader.paths_to_tensor(test_images, (32,32)).astype('float32')

##########################  [SOIL SOYBEANS WEEDS]   ############################
# print(train_images[1500])
# print(tr_labels[1500])
################################################################################

bottleneck = BottleneckModel(pre_model= VGG19, pooling='avg', weights = 'imagenet', input_shape=train_tensors.shape[1:], is_trainable=False)
pre_model = bottleneck.load_bottleneck_model()

classification_model = pre_model
classification_model.add(Dense(256, activation = 'relu'))
classification_model.add(Dropout(0.2))
classification_model.add(Dense(128, activation = 'relu'))
classification_model.add(Dropout(0.2))
classification_model.add(Dense(3, activation='softmax'))
classification_model.summary()

#compiling the classification_model
classification_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#making a checkpointer
checkpointer = ModelCheckpoint(filepath = r'final.hdf5', verbose = 1, save_best_only = True)

#fitting the model
classification_model.fit(train_tensors, tr_labels, batch_size = 100, nb_epoch = 20, validation_split=0.2, callbacks=[checkpointer], shuffle = True)

# saving the classification_model
classification_model.save("model.h5")


classification_model = load_model('model.h5')
score = classification_model.evaluate(test_tensors, ts_labels, verbose = 1)
accuracy = score[1] *100 #score[0] returns loss value, score[1] returns the metrics value (accuracy)
print(r'\n\ test accuracy: ', accuracy)
