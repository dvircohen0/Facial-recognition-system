# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:14:35 2020

@author: דביר
"""


from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import logging
import os
import time
import numpy as np

from utils import take_pic, extract_face, get_embedding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# regist new employee
company_name = input("Enter your company name: ").upper() 
employee_name = input("Enter your full name: ").upper() 
print("Hello,",employee_name, ", please look straight up to the camera")
time.sleep(2)
new_employee_image_path = take_pic()
face_array = extract_face(new_employee_image_path, required_size=(160, 160))
if face_array is not None:
    embedded_face =  get_embedding(face_array)
    in_encoder = Normalizer(norm='l2')
    embedded = expand_dims(embedded_face, axis=0)
    embedded = in_encoder.transform(embedded)
    if not os.path.exists(company_name + '.npz'):
        savez_compressed(company_name + '.npz', embedded_face, employee_name)
    else:
        data = load(company_name + '.npz')
        if employee_name not in data['arr_1']:
            array_tuple = (data['arr_0'],embedded)
            new_data1 = np.vstack(array_tuple)
            array_tuple = (data['arr_1'],employee_name)
            new_data2 = np.vstack(array_tuple)
            savez_compressed(company_name + '.npz', new_data1, new_data2)
        
        
         
        
        
       
        
   
    




# if not os.path.isdir(os.path.join("\database\testing",employee_name)
                     




 


 
    
    
    