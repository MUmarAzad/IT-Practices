#--------------------------Basic Operations with 1D and 2D NumPy Arrays---------------------

import numpy as np
import cv2

arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def array_operations():
    print("1D Array:", arr_1d)
    print("2D Array:\n", arr_2d)
    print("Sum of 1D Array:", np.sum(arr_1d))
    print("Mean of 2D Array:", np.mean(arr_2d))
    print("Transpose of 2D Array:\n", arr_2d.T)

array_operations()

#--------------------------Image Processing with NumPy (Indexing & Slicing)---------------------

def image_processing():
    image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    print("Original Image:\n", image)
    cropped = image[1:4, 1:4]
    print("Cropped Section:\n", cropped)
    inverted_image = 255 - image
    print("Inverted Image:\n", inverted_image)

image_processing()

#--------------------------Augmented Reality Transformation---------------------

def augmented_reality_transformation(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return
    scale_factor = 0.5
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    cv2.imshow("Scaled Image", scaled_image)
    angle = 45
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    cv2.imshow("Rotated Image", rotated_image)
    translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    cv2.imshow("Translated Image", translated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

augmented_reality_transformation('image.jpeg')

#--------------------------Face Detection from Image Arrays---------------------

import cv2
import numpy as np

def face_detection(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        
        cv2.imshow('Face Region', face_region)

        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
        eyes = eyes_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5) 
        
        for (ex, ey, ew, eh) in eyes: 
            eye_region = face_region[ey:ey+eh, ex:ex+ew] 
            cv2.imshow('Eye Region', eye_region) 

    cv2.imshow('Detected Faces', image) 
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

face_detection('image_3.png')
