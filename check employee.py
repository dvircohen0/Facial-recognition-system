import cv2
from time import sleep
from PIL import Image
import PySimpleGUI as sg
from sklearn.preprocessing import Normalizer, OrdinalEncoder
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import sys
from pathlib import Path
from shutil import copyfile
from scipy.spatial import cKDTree
from mtcnn.mtcnn import MTCNN

if getattr(sys, 'frozen', False):
    root_path = sys._MEIPASS
    facenet_path = os.path.join(root_path, 'facenet_keras.h5')
    icon_path = os.path.join(root_path, 'WAI.ico')
    image_path = os.path.join(root_path, 'WAIcheck.png')
    
else:
    facenet_path = 'facenet_keras.h5'
    icon_path = 'WAI.ico'
    image_path = 'WAIcheck.png'
   
# imporove contrast of an image
def improve_contrast_image_using_clahe(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=13.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# check if image contains a face
def check_if_face_in_image(img,detector):
    frame=improve_contrast_image_using_clahe(img)
    faces = detector.detect_faces(frame)
    return faces

#create folder to program
program_folder=os.path.join(Path.home(),"who_am_i")
if not os.path.exists(program_folder):
    os.makedirs(program_folder)

Database_Path =os.path.join(program_folder,"Database.pkl")

   
        
#function that check if a pkl file is a valid databse file
def check_database_valid(Database_path):
    valid=True
    #check if the file path is ok
    if Database_path == None:
            sys.exit()
    #read the pickle file
    try:        
        check_database=pd.read_pickle(Database_path)
        #try to get the columns names from the file
        check_database.columns.values.tolist()
    except:
        if Database_path != os.path.join(program_folder,"Database.pkl"):
            sg.popup_error('This is not a valid Database file', icon=icon_path)
        valid = False
        return valid
    #check if the file gave the excpected columns
    if check_database.columns.values.tolist() == ['Name','ID',"Department","Embedding"]:
        #check if the file is not empty
        if len(check_database) < 1:
            if Database_path != os.path.join(program_folder,"Database.pkl"):
                sg.popup_error('Empty Database!',icon=icon_path)
            valid = False
            return valid
        else:
            valid = True
            return valid
    else:
        if Database_path != os.path.join(program_folder,"Database.pkl"):
            sg.popup_error('This is not a valid Database file',icon=icon_path)
        valid = False
        return valid

Database_loaded=True
while Database_loaded:
    if os.path.isfile(Database_Path):
        if not check_database_valid(Database_Path):
            os.remove(Database_Path)
        else:
            Database=pd.read_pickle(Database_Path)
            Database_loaded=False
    else:
        Database_Path_new=sg.popup_get_file("Select your Database file:",
                              file_types = (('pkl Files', '*.pkl'),),
                              icon=icon_path,
                              initial_folder = program_folder)
        if check_database_valid(Database_Path_new):
            copyfile(Database_Path_new,Database_Path)
            Database_loaded=False
        
Database=pd.read_pickle(Database_Path)
Department_ids =Database.Department.unique().tolist()


sg.change_look_and_feel('DarkBlue2')
sg.popup_animated(image_path,
                  message='Please Wait, Loading…',
                  icon=icon_path)
# load facenet model
tf.keras.backend.clear_session()
facenet = tf.keras.models.load_model(facenet_path)
sg.popup_animated(image_source=None,icon=icon_path)


# sg.theme_previewer()
def get_embedding(face_pixels, facenet):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # 	standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet.predict(samples)
    return yhat[0]


# build Normalize object for  the embedded vector
in_encoder = Normalizer(norm='l2')
# get a list with the name of all departmnts
Departments = [os.path.splitext(i)[0] for i in os.listdir(program_folder) if i.endswith('.pkl')]
# disable capture button until departmnt is selected
dont_allow_pic = True

# build encoder to convert names into labels
out_encoder = OrdinalEncoder()

frame_layout = [[sg.Image(filename="", key="-WEBACM-")]]

graph = sg.Graph((500, 600), (0, 0), (500, 600), key='-G-')

layout = [ [sg.Frame('WHO AM I?',
                    frame_layout)],

          [sg.Text("Department:",
                   justification='right',
                   size=(10, None),
                   pad=(30, 0)),

           sg.Combo(values=Department_ids,
                    size=(15, 1),
                    readonly=True,
                    enable_events=True,
                    key="-Depar-")],

          [sg.Button("check me",
                     bind_return_key=True,
                     pad=(200, 0),
                     disabled=dont_allow_pic)]]
window = sg.Window("check worker",
                   layout,
                   size=(500, 600),
                   location=(350, 0),
                   icon=icon_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = MTCNN()
while True:
    event, values = window.read(timeout=20)
    window['-WEBACM-'](data=cv2.imencode('.png', cap.read()[1])[1].tobytes())

    # enable to take a picture oly if department was selected
    if event == "-Depar-":
        if values["-Depar-"]:
            dont_allow_pic = False
            window.Element("check me").Update(disabled=dont_allow_pic)

    if event == "check me":
        # turn on webcam
        anterior = 0
        while True:
            dont_allow_pic = True
            window.Element("check me").Update(disabled=dont_allow_pic)
            if not cap.isOpened():
                sg.popup("Unable to load camera.", background_color='red',
                         no_titlebar=True,
                         icon=icon_path)
                sleep(5)
                pass
            # Capture frame-by-frame
            ret, frame = cap.read()
            faces=check_if_face_in_image(frame,detector)
            if len(faces) == 0:
                sg.popup_error('Cant detect a face, look straight to the camera',icon=icon_path)
                dont_allow_pic = False
                window.Element("check me").Update(disabled=dont_allow_pic)
                break
            elif len(faces)> 1:
                #sg.popup_error('Detect to many faces, go one by one',icon=icon_path)
                embedded_face=0
                dont_allow_pic = False
                window.Element("check me").Update(disabled=dont_allow_pic)
                break
            else:
            # Draw a rectangle around the faces
                x1, y1, width, height = faces[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_pixels = frame[y1:y2, x1:x2]
                image = Image.fromarray(face_pixels)
                image = image.resize((160, 160))
                face_pixels = np.asarray(image)
                try:
                    # calling get_embedding to get embedded vector of the face
                    embedded_face = get_embedding(face_pixels, facenet)
                    # save the department name from user
                    depart = values["-Depar-"]
                    # window.Element("-Depar-").Update("")
                    #break
                except:
                    continue
                embedded = np.expand_dims(embedded_face, axis=0)
                # Normalize the embedded vector
                embedded = in_encoder.transform(embedded)
                # Read the names from the department database
                Names = Database.Name[Database.Department == depart].values
                full_names = Names
                IDs = Database.ID[Database.Department == depart].values
                IDs = np.array(IDs).astype('str').tolist()
                # Read the embedded vectors from the department database
                embs = Database.Embedding[Database.Department == depart].values
                if len(Names) == 1:
                    if np.linalg.norm(embedded - embs[0]) < 0.4:
                        sg.popup("Access Denied!", background_color='red',
                                 no_titlebar=True,
                                 icon=icon_path)
                        dont_allow_pic = False
                        window.Element("check me").Update(disabled=dont_allow_pic)
                    else:
                        sg.popup("Hello, " + Names[0] + " have a good day!",
                                 background_color='green',
                                 icon=icon_path,
                                 no_titlebar=True)
                else:
                    # convert all embedded vectors to a matrix
                    embs = np.concatenate(embs, axis=0)
                    # encode the labels (the employees names) into numbers
                    out_encoder.fit(Names.reshape(-1, 1))
                    Names = out_encoder.transform(Names.reshape(-1, 1))
                    # find the closest embedding vector
                    dist,index = cKDTree(embs).query(embedded)
                    # convert the prediction from a number to a name
                    predict_name = out_encoder.inverse_transform(Names[index].reshape(-1, 1))
                    if dist < 0.4:
                        # if the score is low deny access
                        sg.popup("Access Denied!", background_color='red',
                                  no_titlebar=True,icon=icon_path)
                        dont_allow_pic = False
                        window.Element("check me").Update(disabled=dont_allow_pic)
                        break
                    else:
                        # if score is high approve access
                        sg.popup("Hello, " + np.squeeze(predict_name), " have a good day!",
                                 background_color='green',
                                 icon=icon_path,
                                 no_titlebar=True)
                        dont_allow_pic = False
                        window.Element("check me").Update(disabled=dont_allow_pic)
                        break
                  
    if event == "Exit" or event == sg.WIN_CLOSED:
        # stop the camera
        cap.release()
        cv2.destroyAllWindows()
        break
window.close()
