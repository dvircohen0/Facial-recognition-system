import cv2
from time import sleep
from PIL import Image
import PySimpleGUI as sg
from sklearn.preprocessing import Normalizer, OrdinalEncoder
import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import sys
import time
from datetime import datetime, timedelta, time
from pathlib import Path
from shutil import copyfile

if getattr(sys, 'frozen', False):
    root_path = sys._MEIPASS
    facenet_path = os.path.join(root_path, 'facenet_keras.h5')
    icon_path = os.path.join(root_path, 'WAI.ico')
    image_path = os.path.join(root_path, 'WAIcheck.png')
    xml_path = os.path.join(root_path, "haarcascade_frontalface_default.xml")
else:
    facenet_path = 'facenet_keras.h5'
    icon_path = 'WAI.ico'
    image_path = 'WAIcheck.png'
    xml_path = "haarcascade_frontalface_default.xml"

#create folder to program
program_folder=os.path.join(Path.home(),"who_am_i")
if not os.path.exists(program_folder):
    os.makedirs(program_folder)
    os.makedirs(os.path.join(program_folder,"Log_files"))

check_log_path = os.path.join(program_folder,"Log_files","check_log.txt")
log_path = os.path.join(program_folder,"Log_files","log.txt")
Database_Path =os.path.join(program_folder,"Database.pkl")

# check if the log files are not empty and databse is valid 
if os.path.isfile(check_log_path):   
    check_log_file = open(check_log_path,"r")
    content = check_log_file.read()
    if  content.isspace() or not content:
        check_log_file.close()
        os.remove(check_log_path)
    else: check_log_file.close()
        
if os.path.isfile(log_path):   
    log_file = open(log_path,"r")
    content = log_file.read()
    if content.isspace() or not content:
        log_file.close()
        os.remove(log_path)
    else: log_file.close()
    
        
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
            sg.popup_error('This is not a valid Database file',
                       icon=icon_path)
        
        valid = False
        return valid
    #check if the file gave the excpected columns
    if check_database.columns.values.tolist() == ['Name',
                                                  'ID',
                                                  "Department",
                                                  "Embedding"]:
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
            sg.popup_error('This is not a valid Database file',
                       icon=icon_path)
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
                  font=('Papyrus'),
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


# pretrained cascade classefier for face
# detecting and crpping from webcam
faceCascade = cv2.CascadeClassifier(xml_path)

# build Normalize object for  the embedded vector
in_encoder = Normalizer(norm='l2')
# get a list with the name of all departmnts
Departments = [os.path.splitext(i)[0] for i in os.listdir(program_folder) if i.endswith('.pkl')]
# disable capture button until departmnt is selected
dont_allow_pic = True
# build a SVC objucr to detect the face from dtaabase
model = SVC(kernel='linear', probability=True)
# build encoder to convert names into labels
out_encoder = OrdinalEncoder()

frame_layout = [[sg.Image(filename="", key="-WEBACM-")]]

graph = sg.Graph((500, 600), (0, 0), (500, 600), key='-G-')

layout = [ [sg.Frame('WHO AM I?',
                    frame_layout,
                    font=('Papyrus', 19))],

          [sg.Text("Department:",
                   justification='right',
                   size=(10, None),
                   pad=(30, 0),
                    font=('Papyrus')),

           sg.Combo(values=Department_ids,
                    size=(15, 1),
                    readonly=True,
                    enable_events=True,
                    key="-Depar-")],

          [sg.Button("check me",
                     bind_return_key=True,
                     pad=(200, 0),
                     font=('Papyrus'),
                     disabled=dont_allow_pic)]]
window = sg.Window("check worker",
                   layout,
                   font=("Papyrus", 12),
                   size=(500, 600),
                   location=(350, 0),
                   icon=icon_path)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
            if not cap.isOpened():
                sg.popup("Unable to load camera.", background_color='red',
                         no_titlebar=True,
                         font=('Papyrus', 15),
                         icon=icon_path)
                sleep(5)
                pass
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in frame
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                # cut a rectangle from the face pic taken by the webcam
                # using opencv haar cascade classefires
                my_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
                # my_face=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                required_size = (160, 160)
                image = Image.fromarray(my_face)
                image = image.resize(required_size)
                # convert to array
                face_pixels = np.asarray(image)
            try:
                # calling get_embedding to get embedded vector of the face
                embedded_face = get_embedding(face_pixels, facenet)

                # sg.popup("pic taken, please wait")
                # save the department name from user
                depart = values["-Depar-"]
                # window.Element("-Depar-").Update("")
                break
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
            if cosine_similarity(embedded, embs[0]) < 0.1:
                sg.popup("Access Denied!", background_color='red',
                         no_titlebar=True,
                         font=('Papyrus', 15),
                         icon=icon_path)
            else:
                sg.popup("Hello, " + Names[0] + " have a good day!",
                         background_color='green',
                         font=('Papyrus', 15),
                         icon=icon_path,
                         no_titlebar=True)
        else:
            # convert all embedded vectors to a matrix
            embs = np.concatenate(embs, axis=0)
            # encode the labels (the employees names) into numbers
            out_encoder.fit(Names.reshape(-1, 1))
            Names = out_encoder.transform(Names.reshape(-1, 1))
            # creat SVC classefier object
            model = SVC(kernel='linear', probability=True)
            # fit the classefier to the embedded vectors and encoded labels
            model.fit(embs, Names.ravel())
            # get the model prediction
            yhat_class = model.predict(embedded)
            # convert the prediction from a number to a name
            predict_name = out_encoder.inverse_transform(yhat_class.reshape(-1, 1))
            predicted_vector = embs[np.where(Names == yhat_class)[0]]
            # compute the cosine distance between the embedded vector
            # to the predict embedded vector
            if cosine_similarity(embedded, predicted_vector) < 0.1:
                # if the score is low deny access
                sg.popup("Access Denied!", background_color='red',
                         font=('Papyrus', 15), no_titlebar=True,icon=icon_path)
            else:
                # if score is high approve access
                sg.popup("Hello, " + np.squeeze(predict_name), " have a good day!",
                         font=('Papyrus', 15),
                         background_color='green',
                         icon=icon_path,
                         no_titlebar=True)
                # get the current time
                current_time = datetime.now().strftime("%H:%M:%S").strip()
                # get the index of the predictet name in thr class
                index = np.where(full_names == np.squeeze(predict_name))[0][0]
                # create a row for the lof file
                row = pd.DataFrame([[full_names[index].replace(" ", "_"),
                                     IDs[index],
                                     values["-Depar-"].replace(" ", "_"),
                                     current_time]])
                # if a log file exist read it
                if os.path.isfile(check_log_path):
                    # check the creation date of the  log file
                    created = os.path.getctime(check_log_path)
                    # convert from timestamp to date
                    date_created = datetime.fromtimestamp(created)
                    # get the time of yesterday midnight
                    midnight = datetime.combine(datetime.today(), time.min)
                    yesterday_midnight = midnight - timedelta(days=1)
                    # check if the file created after yesteday midnight
                    if date_created > yesterday_midnight:                        
                        # read the log file
                        infile = open(check_log_path, 'rb')
                        entering_log=pd.read_csv(infile,delimiter=" ",
                                                 converters={i: str for i in range(3)},
                                                 header=None).fillna('')
                        infile.close()  
                        # the combine the time columns into one column
                        entering_log = pd.concat([entering_log[entering_log.columns[:3]], \
                                                  entering_log[entering_log.columns[3:]].agg(' '.join, axis=1)],
                                                 ignore_index=True, axis=1)
                        # delete multiple spaces in time coulmn
                        entering_log[3] = entering_log[3].replace('\s+', ' ', regex=True)

                        # check ig=f the name id and department already logged in the log file
                        row_cut = row[row.columns[:3]].values
                        # get the location of the logged person in th log file
                        indics = (entering_log[entering_log.columns[:3]] == row_cut).all(1)
                        # check if the person is exist in th log file
                        if indics.any():
                            # add the current time to the alredy logged person
                            entering_log.iloc[indics.idxmax(), 3] += " " + current_time

                        else:
                            # if the person didnt log in today append the row as is
                            entering_log = entering_log.append(row, ignore_index=True)
                       
                else:
                    entering_log = row
                # split the time column into multiple coulumn
                entering_log = pd.concat([entering_log[entering_log.columns[:3]],
                                          entering_log[3].str.split(pat=" ", expand=True)],
                                         ignore_index=True, axis=1)
                # save the log file
                if not os.path.exists(os.path.join(program_folder,"Log_files")):
                    os.makedirs(os.path.join(program_folder,"Log_files"))
                    
                outfile = open(check_log_path, 'w',newline='')
                entering_log.to_csv(outfile,
                                       sep=' ',
                                       index=False,
                                       header=False)
                outfile.close() 
          

    if event == "Exit" or event == sg.WIN_CLOSED:
        # stop the camera
        cap.release()
        cv2.destroyAllWindows()
        break
window.close()

