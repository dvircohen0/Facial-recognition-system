import PySimpleGUI as sg
import os
import pandas as pd
from sklearn.preprocessing import Normalizer
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import sys
from pathlib import Path

sg.change_look_and_feel('DarkBlue2')

if getattr(sys, 'frozen', False):
    root_path = sys._MEIPASS
    facenet_path = os.path.join(root_path, 'facenet_keras.h5')
    icon_path = os.path.join(root_path, 'WAI.ico')
    image_path = os.path.join(root_path, 'WAIcheck.png')
else:
    facenet_path = 'facenet_keras.h5'
    icon_path = 'WAI.ico'
    image_path = 'WAIcheck.png'

sg.popup_animated(image_path,
                  background_color='black',
                  message='Please Wait, Loading…',
                  icon=icon_path )

# load facenet model
tf.keras.backend.clear_session()
facenet = tf.keras.models.load_model(facenet_path)

sg.popup_animated(image_source=None,icon=icon_path )


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


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


dont_allow_save = True
dont_allow_save_del = True
dont_allow_dep_del = True
dont_select_id = True

program_folder=os.path.join(Path.home(),"who_am_i")
if not os.path.exists(program_folder):
    os.makedirs(program_folder)


if not os.path.isfile(os.path.join(program_folder, "Database.pkl")):
    Database = pd.DataFrame(columns=['Name', 'ID',
                                     "Department", "Embedding"])
    Database.to_pickle(os.path.join(program_folder, "Database.pkl"))
else:
    Database = pd.read_pickle(os.path.join(program_folder, "Database.pkl"))

Department_ids = Database.Department.unique().tolist()
id_in_depar = []

tab1_layout = [[sg.Text(key='-EXPAND-0',
                        pad=(0, 10))],

               [sg.Text("Name:",
                        justification='right',
                        size=(11, None)),
                sg.In(size=(25, 1),
                      enable_events=True,
                      key="-Name-"), ],

               [sg.Text("ID:",
                        justification='right',
                        size=(11, None)),
                sg.In(size=(25, 1),
                      enable_events=True,
                      key="-ID-"), ],

               [sg.Text("Department:",
                        justification='right',
                        size=(11, None)),
                sg.Combo(values=Department_ids,
                         size=(25, 1),
                         enable_events=True,
                         key="-Depar-")],

               [sg.Text("Upload Image:",
                        justification='right',
                        size=(11, None)),
                sg.In(size=(25, 1),
                      enable_events=True,
                      key="-Image-PATH-",
                      disabled=True),
                sg.FileBrowse(file_types=(("Image Files", "*.jpg"),
                                          ("PNG Files", "*.png"))), ],

               [sg.Button('Save',
                          bind_return_key=True,
                          size=(20, None),
                          pad=(140, 0),
                          disabled=dont_allow_save)],

               [sg.Text(key='-EXPAND-3',
                        pad=(0, 10))]]

tab2_layout = [[sg.Text(key='-EXPAND-2',
                        pad=(0, 20))],

               [sg.Text("Department:",
                        justification='right',
                        size=(10, None),
                        pad=(20, 0)),
                sg.Combo(values=Department_ids,
                         size=(25, 1),
                         enable_events=True,
                         readonly=True,
                         key="-depardelete-")],

               [sg.Text("ID:",
                        justification='right',
                        size=(10, None),
                        pad=(20, 0)),
                sg.Combo(values=id_in_depar,
                         readonly=True,
                         size=(25, 1),
                         enable_events=True,
                         disabled=dont_select_id,
                         key="-IDdelete-")],

               [sg.Button('Delete from Database',
                          bind_return_key=True,
                          pad=(160, 0),
                          disabled=dont_allow_save_del)], ]

tab3_layout = [[sg.Text(key='-EXPAND-',
                        pad=(0, 10))],

               [sg.Text("Department to Delete:",
                        justification='right',
                        pad=(160, 0))],

               [sg.Combo(values=Department_ids,
                         size=(25, 1),
                         enable_events=True,
                         pad=(120, 0),
                         readonly=True,
                         key="-departocompletedelete-")],

               [sg.Button('Delete Department from Database',
                          pad=(120, 0),
                          bind_return_key=True,
                          disabled=dont_allow_dep_del)], ]

# ----- Full layout -----
layout = [[sg.TabGroup([[sg.Tab('Add New Employee',
                                tab1_layout),
                         sg.Tab('Delete Employee',
                                tab2_layout),
                         sg.Tab('Delete Department',
                                tab3_layout)]])]]

window = sg.Window("Human Resource Managment",
                   layout, icon=icon_path)

# Run the Event Loop
while True:
    event, values = window.read()

    if event == "-departocompletedelete-":
        if values["-departocompletedelete-"]:
            dont_allow_dep_del = False
            window.Element('Delete Department from Database').Update(disabled=dont_allow_dep_del)

    if event == 'Delete Department from Database':
        confirm_delete = sg.popup_ok_cancel('Are you sure?',icon=icon_path)
        if confirm_delete == 'OK':
            Database = Database[Database.Department != values["-departocompletedelete-"]]
            Database.to_pickle(os.path.join(program_folder, "Database.pkl"))
            Department_ids = Database.Department.unique().tolist()
            window.Element("-Depar-").Update(values=Department_ids)
            window.Element("-depardelete-").Update(values=Department_ids)
            window.Element("-departocompletedelete-").Update(values=Department_ids)
            window.Element("-Depar-").Update("")
            window.Element("-depardelete-").Update("")
            window.Element("-departocompletedelete-").Update("")
            sg.popup("Department " + values["-departocompletedelete-"] + " deleted",icon=icon_path)

    if event == "-depardelete-":
        dont_select_id = False
        window.Element("-IDdelete-").Update(disabled=dont_select_id)
        id_in_depar = Database.ID[Database.Department == values["-depardelete-"]].tolist()
        window.Element("-IDdelete-").Update("")
        window.Element("-IDdelete-").Update(values=id_in_depar)
    if event == "-IDdelete-":
        dont_allow_save_del = False
        window.Element("Delete from Database").Update(disabled=dont_allow_save_del)

    if event == 'Delete from Database':
        confirm = sg.popup_ok_cancel('Are you sure?',icon=icon_path)
        if confirm == 'OK':
            Database = Database[Database.ID != values["-IDdelete-"]]
            id_in_depar = Database.ID[Database.Department == values["-depardelete-"]].tolist()
            Database.to_pickle(os.path.join(program_folder, "Database.pkl"))
            window.Element("-IDdelete-").Update(values=id_in_depar)
            window.Element("-IDdelete-").Update("")
            dont_allow_save_del = True
            window.Element("Delete from Database").Update(disabled=dont_allow_save_del)

    if (event == "-Name-" or event == "-ID-" or event == "-Depar-" or event == "-Image-PATH-") and \
            (values["-Name-"] and values["-ID-"] and values["-Depar-"] and values["-Image-PATH-"]):
        dont_allow_save = False
        window.Element("Save").Update(disabled=dont_allow_save)

    if event == 'Save':

        if values["-Depar-"] in Department_ids and \
                values["-ID-"] in Database.ID[Database.Department == values["-Depar-"]].tolist():
            sg.popup("ID ERROR:", "This ID already registered in database",icon=icon_path)
            window.Element("-ID-").Update("")
            continue
        try:
            face_array = extract_face(values["-Image-PATH-"], required_size=(160, 160))
            embedded_face = get_embedding(face_array, facenet)
            in_encoder = Normalizer(norm='l2')
            embedded = np.expand_dims(embedded_face, axis=0)
            embedded = in_encoder.transform(embedded)
        except:
            sg.popup("Image ERROR:", "Try another picture",icon=icon_path)
            window.Element("-Image-PATH-").Update("")
            continue

        row = pd.DataFrame([[values["-Name-"],
                             values["-ID-"],
                             values["-Depar-"],
                             embedded]],
                           columns=['Name', 'ID',
                                    "Department", "Embedding"])
        Database = Database.append(row, ignore_index=True)
        Database.to_pickle(os.path.join(program_folder, "Database.pkl"))
        sg.popup(values["-Name-"] + " added to " + values["-Depar-"],icon=icon_path)
        Department_ids = Database.Department.unique().tolist()
        window.Element("-Name-").Update("")
        window.Element("-ID-").Update("")
        window.Element("-Image-PATH-").Update("")
        window.Element("-depardelete-").Update(values=Department_ids)
        window.Element("-departocompletedelete-").Update(values=Department_ids)
        window.Element("-depardelete-").Update("")
        window.Element("-IDdelete-").Update("")
        dont_allow_save_del = True
        window.Element("Delete from Database").Update(disabled=dont_allow_save_del)
        dont_allow_save = True
        window.Element("Save").Update(disabled=dont_allow_save)

    if event == sg.WIN_CLOSED:
        break
window.close()
