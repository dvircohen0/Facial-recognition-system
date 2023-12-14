# Face Detection and Recognition System
## Introduction
Welcome to the Face Detection and Recognition System project! This repository contains a robust system designed to detect and recognize human faces for the purpose of maintaining a company's employee database. The system is divided into two main components: detection and recognition.
## Features
- **Face Detection**: Add new employee face images to the company's database using the state-of-the-art MTCNN network for accurate face detection.
- **Face Recognition**: Verify and identify personnel by comparing their face images against the existing database using a combination of FaceNet embeddings and SVM classification.

## How It Works
### Detection
The detection component utilizes the pre-trained Multi-task Cascaded Convolutional Networks (MTCNN) to locate faces within images. Once a face is detected, it is cropped and passed through the FaceNet network to generate a corresponding embedded vector.
### Recognition
The recognition component takes the embedded vector of a face and uses a Support Vector Machine (SVM) classifier to determine the closest matching vector in the company's database, effectively identifying the individual.
