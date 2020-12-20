# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:46:36 2020

@author: דביר
"""


from numpy import asarray
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice
import logging
import os
import time
from utils import take_pic, extract_face, get_embedding

company_name = input("Enter your company name: ").upper() 
print("Hello, please look straight up to the camera")
time.sleep(2)
pic_for_test = take_pic()
face_array = extract_face(pic_for_test, required_size=(160, 160))
if face_array is not None:
    embedded_face =  get_embedding(face_array)
    in_encoder = Normalizer(norm='l2')
    embedded = expand_dims(embedded_face, axis=0)
    embedded = in_encoder.transform(embedded)
    
    
data = load(company_name + '.npz')
trainX, trainy = data['arr_0'], data['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
yhat_prob=model.predict_proba(embedded)

# predicttestt
prediction = model.predict(embedded)
print("hello,",data['arr_1'][prediction][0][0],"have a good day!")

