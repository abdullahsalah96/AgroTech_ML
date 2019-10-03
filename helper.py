import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import os

class HelperCallbacks(tf.keras.callbacks.Callback):
    """
    A class that includers helper callback functions
    """
    def __init__(self, property = 'acc', threshold = 0.95):
        self.property = property
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get(property) > self.threshold):
            print('REACHED THRESHOLD SO STOPPING TRAINING')
            self.model.stop_training = True


class HelperImageGenerator():
    """
    A class that generates images from a given directory
    """
    def __init__(self, dir = '/home', target_size = (300, 300), batch_size = 128,  class_mode = 'binary'):
        self.generator = ImageDataGenerator(rescale = 1./255)
        self.dir = dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode

    def generateImages(self):
        return self.generator.flow_from_directory(
        self.dir,
        target_size = self.target_size,
        batch_size = self.batch_size,
        class_mode = self.class_mode
        )


class ImagesLoader():
    def load_images(self, files_path, numOfClasses):
        """
        A funtion that takes the path of the images and returns a numpy array containing the images paths and a numpy array of one hot encoded labels
        """
        data = load_files(files_path, shuffle = False) #load files
        images = np.array(data['filenames']) #load images
        labels = np_utils.to_categorical(np.array(data['target']), numOfClasses) #one hot encoding the labels
        return images, labels

    def path_to_tensor(self, img_path, normalize = True, target_size):
        """
        A funtion that takes the path of the image and converts it into a 4d tensor to be fed to the CNN and normalizes them
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=target_size)
        # convert PIL.Image.Image type to 3D tensor with shape (32, 32, 3)
        x = image.img_to_array(img)
        # print('img shape: ', x.shape[:])
        # convert 3D tensor to 4D tensor with shape (1, 32, 32, 1) and return 4D tensor
        if(normalize):
            return np.expand_dims(x, axis=0)/255
        else:
            return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths, target_size):
        """
        A funtion that takes the path of the images and converts them into a 4d tensor to be fed to the CNN
        """
        list_of_tensors = [self.path_to_tensor(img_path, target_size) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors).astype('float32')

class Annotations():
    def concatenateAnnotationFiles(self, files_path):
        """
        A function that takes the path of the annotation files and returns one csv file
        """
        os.chdir(files_path)
        extension = 'csv'
        all_filenames = [i for i in sorted(glob.glob('*.{}'.format(extension)))]
        #combine all files in the list
        # print(all_filenames)
        combined_csv = pd.concat([pd.read_csv(f, sep = ';') for f in all_filenames ])
        # #export to csv
        return combined_csv

    def getAnnotationsDataframe(self, files_path):
        """
        A function that returns a numpy array that contains x1,x2,y1,y2,class
        """
        annotations = self.concatenateAnnotationFiles(files_path)
        object_class = annotations.drop(['Filename'], axis = 1)
        return object_class

    def resizeBoundingBoxes(self, labels, target_size):
        """
        A function that takes labels df and maps the bounding boxes to the target_size
        """
        resized = labels.copy()
        num_of_images = labels.shape[0]
        new_ROI_X1 = labels['Roi.X1'] * (target_size[0]/labels['Width'])
        new_ROI_Y1 = labels['Roi.Y1'] * (target_size[0]/labels['Height'])
        new_ROI_X2 = labels['Roi.X2'] * (target_size[0]/labels['Width'])
        new_ROI_Y2 = labels['Roi.Y2'] * (target_size[0]/labels['Height'])
        resized['Roi.X1'] = new_ROI_X1
        resized['Roi.Y1'] = new_ROI_Y1
        resized['Roi.X2'] = new_ROI_X2
        resized['Roi.Y2'] = new_ROI_Y2
        return resized

    def getLabels(self, df):
        """
        A function that takes a df and returns a np array with the labels
        """
        labels = df.copy()
        labels = labels.drop(['Width', 'Height'], axis=1)
        labels = np.array(labels)
        labels = np.int_(labels)
        return labels

    def getBoundingBoxes(self, df):
        """
        A function that gets the bounding box from dataframe
        """
        labels = df
        labels = labels.drop(['Width', 'Height', 'ClassId'], axis=1)
        labels = np.array(labels)
        labels = np.int_(labels)
        return labels

    def getTestingBoundingBoxes(self, df):
        """
        A function that gets the bounding box from dataframe
        """
        labels = df
        labels = labels.drop(['Width', 'Height'], axis=1)
        labels = np.array(labels)
        labels = np.int_(labels)
        return labels

    def oneHotEncode(self, labels, size):
        """"
        A function that one hot encodes the labels
        """
        x_onehot = np.identity(size[1])[labels]
        return x_onehot
