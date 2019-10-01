from keras.models import model_from_json
from keras.models import load_model
from helper import ImagesLoader, LabelsLoader
import numpy as np
import cv2

im = ImagesLoader()
ann = LabelsLoader()

class Prediction():
    def get_prediction(model, img_path):
        """
        A funtion that takes the model and the path of the image to be predicted and returns the prediction
        """
        image = im.path_to_tensor(img_path)
        prediction = model.predict(image)
        return prediction

    def get_ROI_prediction(model, img_path):
        """
        A funtion that takes the model and the path of the image and returns mapped prediction
        """
        prediction = self.get_prediction(model, img_path)
        x1 = int(prediction[0][0]*size[0]/32)
        y1 = int(prediction[0][1]*size[1]/32)
        x2 = int(prediction[0][2]*size[0]/32)
        y2 = int(prediction[0][3]*size[1]/32)
        return x1, y1, x2, y2

class Model():
    def get_model(model_path):
        loaded_model = load_model(model_path)
