from watson_developer_cloud import VisualRecognitionV3
import json
import datetime
import cv2
import numpy as np
import threading

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='r5evflSTjmf4aBZJ9uevuyxiDR4w6lld8AwCOMuEq2v5')

# prediction = visual_recognition.classify(images_file = '/home/abdullahsalah96/Testing/Weeds/1177.jpg',threshold= '0.6', classifier_ids = 'Weedclassifier_510344813')
with open('/home/abdullahsalah96/Testing/Weeds/1177.jpg', 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
        threshold='0.6',
	classifier_ids='Weedclassifier_510344813').get_result()
print(json.dumps(classes, indent=2))

class Model():
    def __init__(self, version, iam_apikey, classifier_ids):
        self.version = version
        self.iam_apikey = iam_apikey
        self.visual_recognition = VisualRecognitionV3(version = '2018-03-19', iam_apikey='r5evflSTjmf4aBZJ9uevuyxiDR4w6lld8AwCOMuEq2v5')
        self.classifier_ids = classifier_ids
        self.t1 = threading.Thread(target = self.run_thread)
        self.t1.daemon = True
        self.start = True
        self.img_path = None
        self.threshold = None
        self.end_predicting = False

    def get_json_prediction(self, img_path, threshold):
        with open(img_path, 'rb') as images_file:
            classes = self.visual_recognition.classify(
                images_file,
                threshold=str(threshold),
                classifier_ids=self.classifier_ids).get_result()
        prediction = json.dumps(classes, indent=2)
        return(json.loads(prediction))

    def get_classes(self, prediction):
        classes = prediction['images'][0]['classifiers'][0]['classes'][0]['class']
        return classes

    def predict(self, img_path, threshold):
        # if(self.start):
        #     self.img_path = img_path
        #     self.threshold = threshold
        #     self.start = False
        #     self.t1.start()
        prediction = self.get_json_prediction(img_path, threshold)
        p = self.get_classes(prediction)
        return(p)

    def run_thread(self):
        timer = threading.Timer(0.2, self.timer_prediction)
        timer.start()

    def timer_prediction(self):
        prediction = self.get_json_prediction(self.img_path, self.threshold)
        p = self.get_classes(prediction)
        print(p)

    def stopPrediction(self):
        self.end_predicting = True


# m = Model(version='2016-05-20', iam_apikey= 'Fs1AI_0z05aKN_nQw5Fw6iEYZ-2TWRv0scLu5vbellgT', classifier_ids='Weedclassifier_510344813')
# cap = cv2.VideoCapture()
# cap.open(0)
# while(cv2.waitKey(1)!=27):
#     ret, img = cap.read()
#     cv2.imshow('cap', img)
#     cv2.imwrite('a.jpg', img)
#     m.predict('a.jpg', 0.6)
#
# m.stopPrediction()


# m = Model(version='2016-05-20', iam_apikey= 'r5evflSTjmf4aBZJ9uevuyxiDR4w6lld8AwCOMuEq2v5', classifier_ids='Weedclassifier_510344813')
# # print(datetime.datetime.now().time())
# prediction = m.predict('/home/abdullahsalah96/Testing/Weeds/1177.jpg', 0.6)
# # print(datetime.datetime.now().time())
# # prediction2 = m.predict('/home/abdullahsalah96/Downloads/Cats_test/kitten-tabby.jpg', 0.6)
# # print(datetime.datetime.now().time())
# print(prediction)
