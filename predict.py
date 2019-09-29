from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model
# from train import loaded_model
# from train import test_tensors, test_labels
from helper import ImagesLoader, LabelsLoader
import numpy as np
import cv2

im = ImagesLoader()
ann = LabelsLoader()

def predict_class(model, img_path):
    """
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    image = im.path_to_tensor(img_path)
    prediction = model.predict(image)
    return prediction
    # return(np.argmax(prediction))

# # load json and create model
# json_file = open('/home/abdullahsalah96/Traffic Signs classifier/classification_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# # load weights into new model
# loaded_model.load_weights("/home/abdullahsalah96/Traffic Signs classifier/classification_model.h5")
# print("Loaded model from disk")
# # loaded_model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrics = ['accuracy'])

test_images, ts_labels = im.load_images(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images", 43)
test_tensors = im.paths_to_tensor(test_images).astype('float32')
#getting testing labels
testing_annotations = ann.getAnnotationsDataframe(r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Test_Images/GTSRB/Final_Test/Annotations")
resized_testing_annotations = ann.resizeBoundingBoxes(testing_annotations, (32,32))
testing_labels = ann.getLabels(resized_testing_annotations)
testing_bounding_boxes = ann.getTestingBoundingBoxes(resized_testing_annotations)

loaded_model = load_model('/home/abdullahsalah96/Traffic Signs classifier/model.h5')
# prediction = predict_class(loaded_model, r"/home/abdullahsalah96/Traffic Signs classifier/testing_images/STOP_sign.jpg")
prediction = predict_class(loaded_model, r"/home/abdullahsalah96/Traffic Signs classifier/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/Test/00001.ppm")
print(prediction)
# x1 = prediction[0][0]*400/32
# y1 = prediction[0][1]*400/32
# x2 = prediction[0][2]*400/32
# y2 = prediction[0][3]*400/32
# arr = [x1, y1, x2, y2]
# print(arr)
print(testing_bounding_boxes)

score = loaded_model.evaluate(test_tensors, testing_bounding_boxes, verbose = 1)
accuracy = score[1] #score[0] returns loss value, score[1] returns the metrics value (accuracy)
print(r'\n\Loss: ', accuracy)
