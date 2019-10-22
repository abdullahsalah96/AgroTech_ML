from keras.models import model_from_json
from keras.models import load_model
from helper import ImagesLoader, Annotations
import numpy as np
import cv2

im = ImagesLoader()
classes = { 0: 'soil', 1: 'soybeans', 2: 'weed'}

class Prediction():
    def get_prediction(self, model, img_path):
        """
        A funtion that takes the model and the path of the image to be predicted and returns the prediction
        """
        image = im.path_to_tensor(img_path, True, (32,32))
        prediction = model.predict(image)
        return classes[np.argmax(prediction)]

    def get_ROI_prediction(self, model, img_path):
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
    def get_model(self, model_path):
        """
        a fucntion that takes .h5 model path and returns the loaded model
        """
        loaded_model = load_model(model_path)
        return loaded_model

m = Model()
model = m.get_model('/home/abdullahsalah96/IBM/ML/AgroTech/model.h5')

p = Prediction()
prediction = p.get_prediction(model,'/home/abdullahsalah96/IBM/Dataset/Testing/Soybeans/760.jpg')
print('PREDICTION: ', prediction)
