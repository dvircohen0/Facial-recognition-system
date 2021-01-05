from keras.models import load_model
import numpy as np
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import logging
import os
import cv2
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tkinter import *
from tkinter import ttk
import tkinter as tk 
from PIL import ImageTk, Image
from tkinter import filedialog


"""
This function is the GUI for the take employee picyure system
"""

def take_a_pic():
    #creating global variable
    global last_frame 
    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    global cap
    cap = cv2.VideoCapture(0)
    
    def callback(selection):
        global select
        select = selection
    
    
    def saveImage():
        global img
        img = last_frame
        cv2.imwrite("capturedFrame.jpg",img)
        
    def show_vid(): 
        if not cap.isOpened():
            print("cant open the camera")
        flag, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if flag is None:
            print("Major error!")
        elif flag:
            global last_frame
            last_frame = frame.copy()
    
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    
    root=tk.Tk()
    department_var=StringVar()
    lmain = tk.Label(master=root)
    lmain.grid(column=0, rowspan=4, padx=5, pady=5)
    root.title("click to take a picture")
    clicker = Button(root, text="click to take a picture", command=saveImage)
    clicker.grid(row=5, column=0)
    file_list=[]
    # os.chdir()
    for file in glob.glob("*.npz"):
        file_list.append(file.split('.')[0])
    
    variable = StringVar(root)
    variable.set("select department") # default value
    w = OptionMenu(root, variable, *file_list, command=callback)
    w.grid(row=6, column=0)
    show_vid()
    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly
    cap.release()
    return select
        

"""
This function is the GUI for the new employee register system
"""

def rgister_new_employee():

    root = Tk()
    root.title('Rgister New Employee')
    name_var=StringVar()
    department_var=StringVar()
    image_url_var=StringVar()
    
    
    def save():
        global employe_name, employe_depar
        employe_name = name_var.get()
        employe_depar = department_var.get()
        data.append([employe_name,employe_depar,my_image])
        # root.destroy()
        
    def open():
        global my_image
        root.filename = filedialog.askopenfilename(
        initialdir=r"C:\Users\דביר\face detection deep\5-celebrity-faces-dataset\train",
        title="Select an image",
        filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
        my_image=root.filename
        image_name_label = Label(root, text=root.filename)
        image_name_label.grid(row=5, column=1)
    
       
    f_name = Entry(root,textvariable = name_var, width=30)
    f_name.grid(row=0, column =1, padx=20)
    data=[]
    
    departments_list=[]
    for file in glob.glob("*.npz"):
        departments_list.append(file.split('.')[0])
    
    d_name = ttk.Combobox(root, textvariable=department_var)
    d_name.grid(row=1, column =1, padx=20)
    d_name['values'] = departments_list
    
    
    f_name_label = Label(root, text="Full Name")
    f_name_label.grid(row=0, column=0)
    
    f_name_label = Label(root, text="Department")
    f_name_label.grid(row=1, column=0)
    
    my_btn = Button(root, text="Browse Image", command=open)
    my_btn.grid(row=5, column=0)
    
    submit = Button(root, text="Save", command=save)
    submit.grid(row=6, column=0, columnspan=2, pady=10, padx=10, ipadx=100)

    root.mainloop()
    return data


"""
This function gets a path to an inage and a desired dimensions
and return a face cropped from the image at the desired dimensions
using the MTCNN model
"""

def extract_face(filename, required_size=(160, 160)):
	print("extracting face...")
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
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
	face_array = asarray(image)
	return face_array


"""
This function gets a cropped face image and the FaceNet model
as a input and return the embedded vector of the face using
the model 
"""

def get_embedding(face_pixels, facenet):
	print("creating embedded vector...")
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
# 	standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	yhat = facenet(samples)
	return yhat[0]

"""
This function gets an embedded vector, and an array of embedded vectors as 
an input and return the lowest euclidean distance of the vectors in the array
from the embedded vector 
"""
def auclidaian_distance(embedded_vector, embedded_database):
    dist=[]
    #compute the euclidean distance between the embedded vector
    for i in range(len(embedded_database)):
        dist.append( np.linalg.norm(embedded_vector-embedded_database[i]))
        #return the euclidean distance
        lowest_value = min(dist)
    return lowest_value

"""
This function loads the FaceNet model
"""

def load_facenet():
    print('loading Facenet...')
    facenet = load_model('facenet_keras.h5')
    print('Facenet loaded!')
    return facenet
