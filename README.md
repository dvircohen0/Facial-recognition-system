# face-detection-and-recognition-system


This project consists of two parts: 

# Detection
Adding an employee's face image to the company's database.
The face detection is performed by using the pre-trained MTCNN network.
then we cropping the face image and converting it to an embedded vector by using the FaceNet network.

# Recognition
Checking if a person exists in the company's database and identify him.
We take a face image of a person and convert it to an embedded vector, using the SVM classifier we check the vector closest to it that is in the company database.
