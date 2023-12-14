# Face Detectionand Recognition System

This project consists of two parts: 

## Detection
Adding an employee's face image to the company's database.
The face detection is performed by using the pre-trained MTCNN network.
we converting the cropped image to an embedded vector by using the FaceNet network.

## Recognition
Checking if a person exists in the company's database and identify him.
We take a face image of a person and convert it to an embedded vector, using the SVM classifier we check the vector closest to it that is in the company database.


# Face Detection and Recognition System
## Introduction
Welcome to the Face Detection and Recognition System project! This repository contains a robust system designed to detect and r ecognize human faces for the purpose of maintaining a company's employee database. The system is divided into two main componen ts: detection and recognition.
## Features
- **Face Detection**: Add new employee face images to the company's database using the state-of-the-art MTCNN network for accur ate face detection.
- **Face Recognition**: Verify and identify personnel by comparing their face images against the existing database using a comb ination of FaceNet embeddings and SVM classification.

## How It Works
### Detection
The detection component utilizes the pre-trained Multi-task Cascaded Convolutional Networks (MTCNN) to locate faces within imag es. Once a face is detected, it is cropped and passed through the FaceNet network to generate a corresponding embedded vector. ### Recognition
The recognition component takes the embedded vector of a face and uses a Support Vector Machine (SVM) classifier to determine t he closest matching vector in the company's database, effectively identifying the individual.
