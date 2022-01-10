import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
import winsound as sd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

face_classifier = cv2.CascadeClassifier(r'C:/Users/hoya9/Desktop/Face_Classification/haarcascade_frontalface_default.xml')
classifier = load_model(r'C:/Users/hoya9/Desktop/Face_Classification/Emotion_vgg16.h5')
model = load_model('C:/Users/hoya9/Desktop/LSTM_Help/model/best-model_LSTM.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
actions = np.array(['nothing', 'draw_hand', 'fist', 'help'])

no_sequences = 30
sequence_length = 30
help_count = 0

label_map = {label:num for num, label in enumerate(actions)}

def beepsound():
    fr = 1700    # range : 37 ~ 32767
    du = 75     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def draw_landmarks(images, results):
    mp_drawing.draw_landmarks(images, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(images, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(images, results):
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(images, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # Draw right hand connections    
    mp_drawing.draw_landmarks(images, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    
def hand_extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

colors = [(245, 117, 16), (117, 245, 16), (0, 255, 255), (0, 0, 255)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]} : {round(prob*100, 0)}%', (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
predictions = []
label = []
threshold = 0.6

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = hand_extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1: 
                sentence = sentence[-1:]
                
            image = cv2.flip(image,1)

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x,y)                
            else:
                label = 'No Face Found'
        
        cv2.putText(image, f'{round(max(preds)*100, 1)}%', (503,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (0,0), (160, 40), (0, 0, 0), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (500, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, ''.join(label), (503,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, f'Danger: {help_count}', (243,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if (sentence == ['help']) and (label == 'Sad'):
            help_count += 1
            if help_count > 10:
                cv2.putText(image, 'SOS', (230,240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                beepsound()
                help_count = 0


        # Show to screen
        cv2.imshow('OpenCV Helper', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()