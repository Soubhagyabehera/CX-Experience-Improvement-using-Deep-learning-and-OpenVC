import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.utils import get_file
import argparse
import os
import cvlib as cv
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                      cache_subdir="pre-trained", cache_dir=os.getcwd())

# load model
model = load_model(model_path)
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
classes = ['man','woman']
# Select video or webcam feed
left_counter = 0  # counter for left movement
right_counter = 0  # counter for right movement

th_value = 5  # changeable threshold value


def thresholding(value):  # function to threshold and give either left or right
    global left_counter
    global right_counter

    if (value <= 54):  # check the parameter is less than equal or greater than range to
        left_counter = left_counter + 1  # increment left counter

        if (left_counter > th_value):  # if left counter is greater than threshold value
            print('RIGHT')  # the eye is left
            left_counter = 0  # reset the counter

    elif (value >= 54):  # same procedure for right eye
        right_counter = right_counter + 1

        if (right_counter > th_value):
            print('LEFT')
            right_counter = 0
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()
    # apply face detection
    face, confidence = cv.detect_face(bgr_image)

    print(face)
    print(confidence)
    # loop through detected faces
    frame=bgr_image
    cv2.line(frame, (320, 0), (320, 480), (0, 200, 0), 2)
    cv2.line(frame, (0, 200), (640, 200), (0, 200, 0), 2)
    col = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pupilFrame = frame
    clahe = frame
    blur = frame
    edges = frame
    eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
    detected = eyes.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in detected:  # similar to face detection
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)  # draw rectangle around eyes
        cv2.line(frame, (x, y), ((x + w, y + h)), (0, 0, 255), 1)  # draw cross
        cv2.line(frame, (x + w, y), ((x, y + h)), (0, 0, 255), 1)
        pupilFrame = cv2.equalizeHist(
            frame[y + int(h * .25):(y + h), x:(x + w)])  # using histogram equalization of better image.
        cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # set grid size
        clahe = cl1.apply(pupilFrame)  # clahe
        blur = cv2.medianBlur(clahe, 7)  # median blur
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=7,
                                   maxRadius=21)  # houghcircles
        if circles is not None:  # if atleast 1 is detected
            circles = np.round(circles[0, :]).astype("int")  # change float to integer
            print('integer', circles)
            for (x, y, r) in circles:
                cv2.circle(pupilFrame, (x, y), r, (0, 255, 255), 2)
                cv2.rectangle(pupilFrame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                # set thresholds
                thresholding(x)
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(bgr_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(bgr_image[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(bgr_image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image,color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45,1,3)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    cv2.imshow('image4', pupilFrame)
    cv2.imshow('clahe', clahe)
    cv2.imshow('blur', blur)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()