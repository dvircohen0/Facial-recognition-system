from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import logging
import os
import numpy as np
from utils import  extract_face, get_embedding,rgister_new_employee, load_facenet

#ignoring errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#loading facenet model
facenet = load_facenet()

#open GUI to enter new employees data
employees_data = rgister_new_employee()

for employee in employees_data:
    #getting employee name
    employee_name = employee[0]
    print(employee_name)
    #getting employee department
    company_name = employee[1].upper()
    #getting employee image path
    image_path = employee[2]

    while True:
        # if department already exist load its data
        if  os.path.exists(company_name + '.npz'):
            data = load(company_name + '.npz')
            # if employee is already registered in the system
            if employee_name in data['arr_1']:
                print("employee is allready in database")
                break
        # getting the cropped face from the image
        try:
            face_array = extract_face(image_path, required_size=(160, 160))
        #calling get_embedding to get embedded vector of the face
            embedded_face =  get_embedding(face_array, facenet)
            #Normalize the embedded vector
            in_encoder = Normalizer(norm='l2')
            embedded = expand_dims(embedded_face, axis=0)
            embedded = in_encoder.transform(embedded)
            #if  department not exsit save the embedded vectors in a new file
            if not os.path.exists(company_name + '.npz'):
                savez_compressed(company_name + '.npz', embedded, employee_name)
                break
            #add a new embedded vector to exsiting deparment file
            array_tuple = (data['arr_0'],embedded)
            new_data1 = np.vstack(array_tuple)
            array_tuple = (data['arr_1'],employee_name)
            new_data2 = np.vstack(array_tuple)
            savez_compressed(company_name + '.npz', new_data1, new_data2)
            break
        except: 
            print(employee_name + ", please try another pic")
            break
