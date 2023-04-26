from flask import Flask, render_template, request, redirect, url_for, session, Response
import re
import mysql.connector
import os
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd
import IPython
from IPython.display import Audio
import time
from time import sleep
from pathlib import Path
import threading
from flaskext.mysql import MySQL
import csv

app = Flask(__name__)
app.secret_key = 'neverfall'



_total = 0 
_sittostand = 0

login = False
cam_status = False
# Connect to the MySQL database
#total,time = None
cnx = None

@app.before_request
def initDB():
    global cnx
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'nerver fall'

    cnx = mysql.connector.connect(

            host="localhost",
            user="root",
            password="",
            database="never fall"
        ) 

@app.route('/')

@app.route('/login', methods =['GET', 'POST'])
def login():

    global login
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'phone' in request.form :

        username = request.form['username']
        phone = request.form['phone']
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = % s AND phone = % s', (username, phone))
        cursor.execute('SELECT * FROM users WHERE username = %s AND phone = %s', (username, phone))
        account = cursor.fetchone()
        id = account["Id"]
        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()
        
        cursor.execute('SELECT * FROM Checklist WHERE Id = %s order by time DESC', (id,))
        checklist = cursor.fetchone()

        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        cursor.execute('SELECT * FROM test WHERE Id = %s order by time DESC' , (id,))
        tug = cursor.fetchone()

        row = cursor.fetchone()
        while row is not None:
            print(row) 
            row = cursor.fetchone()
        
        risk = ''
        if tug['result'] == 'High' and checklist['sum'] >= 7 :
           risk = 'High risk of falling'
        elif tug['result'] == 'High' and checklist['sum'] <= 7 :
            risk = 'Risk of falling'
        elif tug['result'] == 'Low' and checklist['sum'] <= 7 :
            risk = 'Good'


        if account:
            session['loggedin'] = True
            session['username'] = account['username']
            msg = 'Logged in successfully !'
            login = True
            return render_template('index.html', msg = msg ,risk = risk, status = login , account = account, checklist = checklist, tug = tug)

        else:
            msg = 'Incorrect username / phone !'
    return render_template('login.html', msg = msg, status = login)

@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('username', None)
   return redirect(url_for('login'))    

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'phone' in request.form and 'gender' in request.form and 'age' in request.form and 'height' in request.form and 'weight' in request.form:
        username = request.form['username']
        phone = request.form['phone']
        gender = request.form['gender']
        age= request.form['age']
        height = request.form['height']
        weight = request.form['weight']
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = % s', (username, ))
        #cursor.execute('SELECT * FROM users WHERE username = %s ', (username))
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        
        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'name must contain only characters and numbers !'
        else:
            cursor.execute('INSERT INTO users VALUES (NULL,%s, %s, %s, %s, %s, %s)', (username, phone, gender,age, height, weight))
            cnx.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

@app.route("/index")
def index():
    if 'loggedin' in session:
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = % s AND phone = % s', (username, phone))
        cursor.execute('SELECT * FROM users WHERE username = %s', (session['username'],))
        account = cursor.fetchone()
        id = account["Id"]
        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()
        
        cursor.execute('SELECT * FROM Checklist WHERE Id = %s order by time DESC', (id,))
        checklist = cursor.fetchone()

        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        cursor.execute('SELECT * FROM test WHERE Id = %s order by time DESC' , (id,))
        tug = cursor.fetchone()

        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        risk = 'Default'
        if tug['result'] == 'High' and checklist['sum'] >= 7 :
            risk = 'High risk of falling'
        elif tug['result'] == 'High' and checklist['sum'] <= 7 :
            risk = 'Risk of falling'
        elif tug['result'] == 'Low' and checklist['sum'] <= 7 :
            risk = 'Good'
        else:
            risk = 'Default'
            tug = 'Default'
            checklist =  'Default'
    msg = ''
    print(risk)
    return render_template('index.html', msg = msg ,risk = risk, status = login , account = account, checklist = checklist, tug = tug)
    #return redirect(url_for('login'))

@app.route("/display")
def display():
    if 'loggedin' in session:
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = % s', (session['username'], ))
        cursor.execute('SELECT * FROM users WHERE username = %s', (session['username'], ))
        account = cursor.fetchone()   
        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        return render_template("display.html", account = account , )
    return redirect(url_for('login'))

@app.route("/update", methods =['GET', 'POST'])
def update():
    #print(request.form)
    msg = ''
    if 'loggedin' in session:
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
        #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
        cursor.execute('SELECT * FROM users WHERE username = %s', (session['username'], ))
        account = cursor.fetchone()
        if request.method == 'POST':
            id = account["Id"]
            print(id)
            username = request.form['username']
            phone = request.form['phone']
            gender = request.form['gender']
            age= request.form['age']
            height = request.form['height']
            weight = request.form['weight']
            #cursor = cnx.cursor(dictionary=True)
            #cursor.execute('SELECT * FROM users WHERE username = % s', (username, ))
            #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
            cursor.execute('SELECT * FROM users WHERE Id = %s',(id,))
            account = cursor.fetchone()
            if account is not None:
                print("this one come in")
                cursor.execute('UPDATE users SET username = %s, phone = %s, gender = %s, age = %s, height = %s, weight = %s WHERE Id = %s', (username, phone, gender, age, height, weight,id))
                cnx.commit()
                msg = 'You have successfully updated!'
            elif account:
                msg = 'Account already exists !'
            elif not re.match(r'[A-Za-z0-9]+', username):
                msg = 'name must contain only characters and numbers !'
            else:
                msg = 'Account does not exist!'
            cursor.close()
        elif request.method == 'GET':
            msg = 'Please fill out the form !'
        return render_template("update.html", msg = msg)
    #return redirect(url_for('login'))

# @app.before_request
@app.route('/test_data', methods = ['POST' , 'GET'])
def updata():
    if 'loggedin' in session:
        cursor = cnx.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (session['username'],))
        account = cursor.fetchone()
        cnx.commit()
        if request.method == 'POST':
            result = 'risk'
            if _total>12:
                result == "High"
            elif _total<=12:
                result == "Low"
            print("This is result in updatedata : " , result)
            cursor.execute('INSERT INTO test VALUES (%s,%s,%s,%s,%s)', (None,account["Id"],_total,_sittostand,result))
            cnx.commit()

        return render_template("test.html" , total = _total , sittostand = _sittostand)

@app.route("/gen")                      
#def gen(camera):
def gen():
    counter = 0
    stage = None
    i=0
    start_time = time.time()
    time_spare = 0
    sit_count = []
    cap = cv2.VideoCapture(1)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ =  image.shape
                image = cv2.resize(image, (int(image_width * (500 / image_height)), 700))
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    # Calculate angle
                    angle = calculate_angle(hip, knee, wrist)
                    # Visualize angle
                    cv2.putText(image, 
                                tuple(np.multiply(knee, [1080, 720]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    cv2.putText(image, 
                                tuple(np.multiply(hip, [1080, 720]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    # Curl counter logic
                
                    if angle > 30:
                        stage = "sit"
                        sit_count.append(1)
                        print("hip_sit: ",hip)
                        print("knee_sit: ",knee)
                    if  stage =='sit' and angle < 15 :
                        stage="stand"
                        counter = time.time() - start_time
                        counter = float("{0:.2f}".format(counter))
                        print("counter:",counter,"seconds")
                        print("Sit to stand",float("{0:.2f}".format(counter-time_spare)),"seconds")
                        time_spare = counter
                        print("hip_sit: ",hip)
                        print("knee_sit: ",knee)
                        sit_count.append(0) 
                        #print("sit_count",sit_count) 
                        sittostand = float("{0:.2f}".format(counter-time_spare))
                    
                except:
                    pass
                total = float("{0:.2f}".format(time.time() - start_time))
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (255,102,51), -1)
                # Rep data
                cv2.putText(image, str(total), (745,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Time', (750,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(image, (120,650), (1000,700), (153,204,255), -1)
                cv2.putText(image, 'Please back to index page to see the result', (150,680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )           
                ret,buffer=cv2.imencode('.jpg',image)
                frame=buffer.tobytes()
                    #process_video_stream(frame)
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
                    break

                elif sit_count[-4:] == [0,1,1,1]:
                    print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
                    break
    
        if request.method == 'POST' and 'username' in request.form and 'phone' in request.formand and 'Id' in request.form and 'gender' in request.form and 'age' in request.form and 'height' in request.form and 'weight' in request.form and 'total' in request.form and 'sittostand' in request.form and 'result' in request.form:
            username = request.form['username']
            Id = request.form['Id']
            phone = request.form['phone']
            gender = request.form['gender']
            age= request.form['age']
            height = request.form['height']
            weight = request.form['weight']
            total = request.form['total']
            sittostand = request.form['sittostand']
            result = request.form['result']
            cursor = cnx.cursor(dictionary=True)
            cursor.execute('SELECT * FROM user WHERE username = % s', (username, ))
            #account = cursor.fetchone()
            cursor.execute('INSERT INTO user VALUES (%s,,%s,%s, %s, %s, %s, %s, %s)', (Id,username,phone,gender,age,height,weight,total,sittostand,result))
            cnx.commit()
        return render_template("display.html" , account = "" , total = 0 , sittostand = "")
    cap.release() 
    cv2.destroyAllWindows() 
    return image,total,sittostand              


        
def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
    
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
    
        if angle >180.0:
            angle = 360-angle   
        return angle

@app.route("/video/<data>" , methods = ['POST' , 'GET'])
# @app.before_request
# route the neverfall.py
def process_video_stream():
    global _total , _sittostand
    now = datetime.now().time()
    now = (str(now).replace(".", "_").replace(":", "_"))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #open camera
    timer = 1
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0
    stage = None
    i=0
    start_time = time.time()
    time_spare = 0
    #test = True
    #prev_frame_time = 0
    #new_frame_time = 0
    sit_count = []
    #time = datetime.datetime.now().strftime("%H:%M:%S")
    #global total,sittostand
    #total = 0.0
    #sittostand = 0.0

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:

            ret, frame = cap.read()
            cv2.imencode('.jpg',frame)
            if not ret:
                break

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ =  image.shape
                image = cv2.resize(image, (int(image_width * (500 / image_height)), 700))
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                    right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y
                    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                    left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y
                    ret, frame = cap.read()
                    # Curl counter logic
                    if right_hand_y < right_shoulder_y and left_hand_y < left_shoulder_y :
                        start_time = time.time() 
                        while timer >= 0:
                            

                            cv2.putText(frame, str(timer), 
                            (200, 250), cv2.FONT_HERSHEY_SIMPLEX,
                            7, (0, 255, 255),
                            4, cv2.LINE_AA)
                            

                            cur = time.time()
                            if cur - start_time >= 1:
                                start_time = cur
                                timer = timer -1

                        else:
                            ret, frame = cap.read()
                            cv2.imencode('.jpg', frame)
                            #start_time = time.time()
                            if knee_y - hip_y < 0.03:
                                stage = "sit"
                                sit_count.append(1)
                                #print("hip_sit: ",hip)
                                #print("knee_sit: ",knee)
                            if  stage =='sit' and knee_y - hip_y >= 0.03 :
                                stage="stand"
                                counter = time.time() - start_time
                                counter = float("{0:.2f}".format(counter))
                                print("counter:",counter,"seconds")
                                print("Sit to stand",float("{0:.2f}".format(counter-time_spare)),"seconds")
                                sittostand = counter - time_spare
                                time_spare = counter
                                sit_count.append(0) 
                                #print("hip_sit: ",hip)
                                #print("knee_sit: ",knee)
                                #print("sit_count",sit_count)
                            
                            
            

                      
                except:
                    pass
                total = float("{0:.2f}".format(time.time() - start_time))
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                # Rep data
                #cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(total), (745,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Time', (750,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                #cv2.createButton("Calculate",updata,None,cv2.QT_PUSH_BUTTON,1)      
                #cv2.putText(image, time, (640,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                #cv2.putText('STAGE', (65,750), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )           
                ret,buffer=cv2.imencode('.jpg',image)
                frame=buffer.tobytes()
                    #process_video_stream(frame)
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
                #cv2.imshow('Mediapipe Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
                    break


                elif sit_count[-4:] == [0,1,1,1]:
                    print("Total",float("{0:.2f}".format(time.time() - start_time)),"seconds")
                    break

                total = float("{0:.2f}".format(time.time() - start_time))
                # print("The end of the processes...")

                cap.release()
                cv2.destroyAllWindows()
                
    #return total,sittostand
    _total = total
    _sittostand = sittostand
    #return redirect(url_for('updata'))

def video():
    global _total , _sittostand
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose  

    cap = cv2.VideoCapture(0)
    #posture.csv = []
    #header = ["time","diff_shoulder","ratio_armR","ratio_armL","diff_hip","ratio_legR","ratio_legL","should meet doctor?"]
    # Initiate a timer
    start_time = None
    end_time = None
    prev_time = 0
    timer_started = None
    # Initiate counters
    sits = 0
    stage = None
    sec = None
    tester = 3
    # Set up Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            cv2.imencode('.jpg',frame)
            if not ret:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert the RGB image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
           
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            except AttributeError:
                continue
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract coordinate values of interest
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y

            right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y

            right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

            left_ankel_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankel_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        
            diff_shoulder = abs(right_shoulder_y-left_shoulder_y)
            ratio_armR = abs(right_shoulder_y-right_elbow_y)/abs(right_elbow_y-right_hand_y)
            ratio_armL = abs(left_shoulder_y-left_elbow_y)/abs(left_elbow_y-left_hand_y)
            diff_hip = abs(right_hip_y-left_hip_y)
            ratio_legR = abs(right_hip_y-right_knee_y)/abs(right_knee_y-right_ankel_y)
            ratio_legL = abs(left_hip_y-left_knee_y)/abs(left_knee_y-left_ankel_y)

            List = [tester,sec,diff_shoulder,ratio_armR,ratio_armL,diff_hip,ratio_legR,ratio_legL]
            # Stage 1: Check if right hand is above right shoulder
            if stage == None:
                if right_hand_y < right_shoulder_y and left_hand_y < left_shoulder_y:
                    curr_time = time.time()
                    stage = 1
                    start_time = time.time() + 5
                    prev_time = start_time
                    timer_started = True 
                    text = "Timer Started"  
                else:
                    text = "Raise both hand above shoulder to start timer"
            # Stage 2: Detecting sits and measuring time
            elif stage == 1 and start_time >= 0:
                if left_knee_y - left_hip_y < 0.03 and right_knee_y - right_hip_y < 0.03 :
                    sits += 1
                    stage = 2
                    end_time = time.time()
                    sec = curr_time - prev_time
                    text = "Sit Detected, Total Sits: " + str(sits) + ", Time: " + str(round(end_time - start_time, 2)) + "s"
                    print("Time",sec,"diff_shoulder",diff_shoulder,"ratio_armR",ratio_armR,"ratio_armL",ratio_armL,"diff_hip",diff_hip,"ratio_legR",ratio_legR,"ratio_legL",ratio_legL)
                    with open('posture.csv', mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(List)
                    csv_file.close()
                else:
                    text = "Stand Straight"
                    sec = curr_time - prev_time
                    print("Time",sec,"diff_shoulder",diff_shoulder,"ratio_armR",ratio_armR,"ratio_armL",ratio_armL,"diff_hip",diff_hip,"ratio_legR",ratio_legR,"ratio_legL",ratio_legL)
                    with open('posture.csv', mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(List)
                    csv_file.close()
            # Display stage text
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if timer_started:
                curr_time = time.time()
                timer = curr_time - prev_time
                if timer < 0:
                    #print(timer)
                    timer = abs(timer)
                    cv2.putText(image, "Countdown: {:.0f}".format(timer), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                    #print(timer)
                elif timer >= 0:
                    cv2.putText(image, "Timer: {:.2f}".format(timer), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                    
            # Display frame
            ret,buffer=cv2.imencode('.jpg',image)
            frame=buffer.tobytes()
            #process_video_stream(frame)
            yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #cv2.imshow('Mediapipe Feed', image)

    cap.release()
    cv2.destroyAllWindows()

@app.route("/neverfall")
def neverfall():
    return Response(video(),mimetype='multipart/x-mixed-replace; boundary=frame')
    


@app.route("/checklist", methods =['GET', 'POST'])
def checklist():
    msg = ''
    risk = ''
    if 'loggedin' in session:
        cursor = cnx.cursor(dictionary=True)
        #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
        #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
        cursor.execute('SELECT * FROM users WHERE username = %s', (session['username'], ))
        account = cursor.fetchone()
        if request.method == 'POST':
            id = account["Id"]
            print("/////////////////")

            fall = request.form.get('fall')
            if fall:
                fall = 1
            else:
                fall = 0

            bendedback = request.form.get('bendedback')
            if bendedback:
                bendedback = 1
            else:
                bendedback = 0

            walkslow = request.form.get('walkslow')
            if walkslow :
                walkslow  = 1
            else:
                walkslow  = 0

            cane= request.form.get('cane')
            if cane:
                cane = 1
            else:
                cane = 0

            medicines = request.form.get('medicines')
            if medicines:
                medicines = 1
            else:
                medicines = 0
            
            #cursor.execute('SELECT * FROM users WHERE username = % s', (username, ))
            #cursor.execute('SELECT * FROM users WHERE username = %s ', (username, ))
            cursor.execute('SELECT * FROM users WHERE Id = %s',(id,))
            account = cursor.fetchone()
            row = cursor.fetchone()
            while row is not None:
                print(row)
                row = cursor.fetchone()

            print("..................")
            print(type(fall))
            
            sum = ((fall*5)+(bendedback+walkslow+cane+medicines)*2)
            if sum >= 7:
                risk = 'high'
            elif sum < 7:
                risk = 'low'
            cursor.execute('INSERT INTO Checklist VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s)', (None,id,fall, bendedback, walkslow, cane, medicines, sum, risk))
            cnx.commit()
            msg = 'Thank you for doing the checklist!'
    return render_template("checklist.html", msg = msg)


if __name__ == '__main__':
    app.run(debug=True)