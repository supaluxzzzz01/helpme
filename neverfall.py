import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd
import IPython
from IPython.display import Audio
import time
from pathlib import Path

def process_video_stream(cap):
    now = datetime.now().time()
    now = (str(now).replace(".", "_").replace(":", "_"))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
    
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
    
        if angle >180.0:
            angle = 360-angle   
        return angle

    #open camera
    #cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0
    stage = None
    i=0
    start_time = time.time()
    time_spare = 0
    test = True
    prev_frame_time = 0
    new_frame_time = 0
    sit_count = []

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ =  image.shape
            image = cv2.resize(image, (int(image_width * (1000 / image_height)), 1000))
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [1080, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Curl counter logic
            
                if angle > 30:
                    stage = "sit"
                    sit_count.append(1)
                if  stage =='sit' and angle < 15 :
                    stage="stand"
                    counter = time.time() - start_time
                    counter = float("{0:.2f}".format(counter))
                    print(counter,"seconds")
                    print("Sit to stand",float("{0:.2f}".format(counter-time_spare)),"seconds")
                    time_spare = counter
                    sit_count.append(0)  
                    
            except:
                pass
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
                break
            #elif sit_count[-2:] == [0,1]:
            #   print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
            #  break
            
        cap.release()
        cv2.destroyAllWindows()
    return image