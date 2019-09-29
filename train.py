# from scipy.misc import imread
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras import metrics
import numpy as np
from glob import glob
from skimage import color
from tqdm import tqdm
import cv2
import os
# from utils import load_images, path_to_tensor, paths_to_tensor
from helper import HelperCallbacks, HelperImageGenerator, ImagesLoader, LabelsLoader

#importing the training and testing images
imageLoader = ImagesLoader()
train_images, tr_labels = imageLoader.load_images(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images", 43)
test_images, ts_labels = imageLoader.load_images(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images", 43)


# train_images, tr_labels = imageLoader.load_images(r"/home/abdullahsalah96/Traffic Signs classifier/BelgiumTSC_Training/Training", 62)
# test_images, ts_labels = imageLoader.load_images(r"/home/abdullahsalah96/Traffic Signs classifier/BelgiumTSC_Testing/Testing", 62)

#converting the training and testing images to 4d tensors to be fed to the convolutional layers
train_tensors = imageLoader.paths_to_tensor(train_images).astype('float32')
test_tensors = imageLoader.paths_to_tensor(test_images).astype('float32')

ann = LabelsLoader()
#getting training labels
training_annotations = ann.getAnnotationsDataframe(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Training_Images/GTSRB/Final_Training/Annotations")
resized_training_annotations = ann.resizeBoundingBoxes(training_annotations, (32,32))
training_labels = ann.getLabels(resized_training_annotations)
training_bounding_boxes = ann.getBoundingBoxes(resized_training_annotations)
training_encoded_bounding_boxes = ann.oneHotEncode(training_bounding_boxes, (32,32))
final_training_bounding_boxes = training_encoded_bounding_boxes.reshape([39209, 128]) #reshape bounding boxes where each 32 elements is a new point

#getting testing labels
testing_annotations = ann.getAnnotationsDataframe(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Test_Images/GTSRB/Final_Test/Annotations")
resized_testing_annotations = ann.resizeBoundingBoxes(testing_annotations, (32,32))
testing_labels = ann.getLabels(resized_testing_annotations)
testing_bounding_boxes = ann.getTestingBoundingBoxes(resized_testing_annotations)

#concatenating bbxes with CLASSES
print(train_images[0])
print("--FINAL CLASSES-- \n")
print(tr_labels[0])
print("--FINAL BOXES-- \n")
print(training_bounding_boxes[0])
print("--FINAL ENCODED BOUDNING BOXES-- \n")
print(training_encoded_bounding_boxes[0])
print("--FINAL FLATTENED BOXES-- \n")
print(final_training_bounding_boxes[0])

print("--FINAL TESTING BOXES")
print(testing_bounding_boxes)


# #################bounding boxes prediction##################
# # classification_model's architecture
# classification_model = Sequential()
# classification_model.add(Conv2D(filters = 16, kernel_size = 2, padding='same', strides = 1, activation = 'relu', input_shape = train_tensors.shape[1:]))
# classification_model.add(Dropout(0.2))
# classification_model.add(MaxPooling2D(pool_size = 2))
# classification_model.add(Conv2D(filters = 32, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(MaxPooling2D(pool_size = 2))
# classification_model.add(Conv2D(filters = 64, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(MaxPooling2D(pool_size = 2))
# classification_model.add(Conv2D(filters = 128, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(MaxPooling2D(pool_size = 2))
# classification_model.add(Conv2D(filters = 256, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(MaxPooling2D(pool_size = 2))
# classification_model.add(Conv2D(filters = 512, kernel_size = 2, padding='same', strides = 1, activation = 'relu'))
# classification_model.add(GlobalAveragePooling2D())
# classification_model.add(Dropout(0.1))
# classification_model.add(Dense(1024, activation='relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(Dense(512, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(Dense(256, activation = 'relu'))
# classification_model.add(Dropout(0.2))
# classification_model.add(Dense(4))
# classification_model.summary()
#
# #compiling the classification_model
# classification_model.compile(loss='mse', optimizer='adam', metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity])
# # #making a checkpointer
# checkpointer = ModelCheckpoint(filepath = r'best_weights\final.hdf5', verbose = 1, save_best_only = True)
# #
# # #fitting the model
# classification_model.fit(train_tensors, training_bounding_boxes, batch_size = 100, nb_epoch = 10, validation_split=0.2, callbacks=[checkpointer], shuffle = True)
#
# # saving the classification_model
# classification_model.save("/home/abdullahsalah96/Traffic Signs classifier/model.h5")
#
# classification_model_json = classification_model.to_json()
# with open("/home/abdullahsalah96/Traffic Signs classifier/classification_model.json", "w") as json_file:
#     json_file.write(classification_model_json)
# classification_model.save_weights("/home/abdullahsalah96/Traffic Signs classifier/classification_model.h5")
# print("Saved classification_model to disk")
#
# score = classification_model.evaluate(test_tensors, testing_bounding_boxes, verbose = 1)
# accuracy = score[1] *100 #score[0] returns loss value, score[1] returns the metrics value (accuracy)
# print(r'\n\nAccuracy score: ', accuracy)
#
# #################bounding boxes prediction##################
