# -*- coding: utf-8 -*-

from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras.models import load_model
import pickle

class SVM_fit : 
    def __init__(self) :
        self.PATH = '/test/'
        self.data = load(self.PATH+'all-other-embeddings_hangul.npz')
        self.trainX, self.trainy = self.data['arr_0'], self.data['arr_1']

        # normalize input vectors
        self.in_encoder = Normalizer(norm='l2')
        self.trainX = self.in_encoder.transform(self.trainX)

        # label encode targets
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(self.trainy)
        self.trainy = self.out_encoder.transform(self.trainy)
        
        self.model = SVC(kernel='linear', C=1.0, probability=True) # C 값 조절 가능
        self.model.fit(self.trainX, self.trainy)

        face_model = {
            "in_encoder":self.in_encoder,
            "out_encoder":self.out_encoder,
            "svm":self.model
        }
        model_path = self.PATH+"svm_docker_test.pkl"
        f = open(model_path, "wb")
        f.write(pickle.dumps(face_model))
        f.close()
        print("저장 완료")



if __name__ == '__main__':
    s = SVM_fit()
