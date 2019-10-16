from watson_developer_cloud import VisualRecognitionV3
import json
import datetime

class Model():
    def __init__(self, version, iam_apikey, classifier_ids):
        self.version = version
        self.iam_apikey = iam_apikey
        self.visual_recognition = VisualRecognitionV3(version = '2016-05-20', iam_apikey='Fs1AI_0z05aKN_nQw5Fw6iEYZ-2TWRv0scLu5vbellgT')
        self.classifier_ids = classifier_ids

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
        prediction = self.get_json_prediction(img_path, threshold)
        p = self.get_classes(prediction)
        return(p)

m = Model(version='2016-05-20', iam_apikey= 'Fs1AI_0z05aKN_nQw5Fw6iEYZ-2TWRv0scLu5vbellgT', classifier_ids='DefaultCustomModel_2007676959')
print(datetime.datetime.now().time())
prediction = m.predict('/home/abdullahsalah96/Testing/Soil/686.jpg', 0.6)
print(datetime.datetime.now().time())
# prediction2 = m.predict('/home/abdullahsalah96/Downloads/Cats_test/kitten-tabby.jpg', 0.6)
# print(datetime.datetime.now().time())
print(prediction)
