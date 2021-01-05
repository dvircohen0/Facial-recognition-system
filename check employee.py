from numpy import load
import numpy as np
from keras.models import load_model
from numpy import expand_dims
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from utils import extract_face, get_embedding,take_a_pic, auclidaian_distance,  load_facenet
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#loading facenet model
facenet = load_facenet()

#open GUI to get image from webcam and enternig deparment name
company_name=take_a_pic()
# getting the cropped face from the image
face_array = extract_face("capturedFrame.jpg", required_size=(160, 160))
    
if face_array is not None:
    #calling get_embedding to get embedded vector of the face
    embedded_face =  get_embedding(face_array,facenet)
    #Normalize the embedded vector
    in_encoder = Normalizer(norm='l2')
    embedded = expand_dims(embedded_face, axis=0)
    embedded = in_encoder.transform(embedded)

# load the depatmrnt embedded vectors
data = load(company_name + '.npz')
# put the embedded vectors in trainX and the labels in trainy 
trainX, trainy = data['arr_0'], data['arr_1']
# encode the labels (the employees names) into numbers
out_encoder = OrdinalEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
# creat SVC classefier object
model = SVC(kernel='linear', probability=True)
# fit the classefier to the embedded vectors and encoded labels
model.fit(trainX, trainy)

# get the model prediction
yhat_class = model.predict(embedded)
# get the model probabilities for each label
yhat_prob = model.predict_proba(embedded)
#convert the prediction from a number to a name
predict_names = out_encoder.inverse_transform(yhat_class.reshape(-1, 1))
#compute the euclidean distance between the embedded vector
# to the predict embedded vector 
if auclidaian_distance(embedded,data['arr_0']) > 1.2 :
    print("Access Denied!")
else:
    print("hello,", np.squeeze(predict_names),"have a good day!")

    





