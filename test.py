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
from pathlib import Path
import threading
from flaskext.mysql import MySQL

id = 1

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

        row = cursor.fetchone()
        while row is not None:
            print(row)
            row = cursor.fetchone()

        if account:
            session['loggedin'] = True
            session['username'] = account['username']
            msg = 'Logged in successfully !'
            login = True
            return render_template('index.html', msg = msg , status = login)

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
    return render_template("index.html")

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
        if request.method == 'POST':
            result = 'risk'
            if _total>12:
                result == "High risk"
            elif _total<=12:
                result == "Low risk"
            print("This is result in updatedata : " , result)
            cursor.execute('INSERT INTO test VALUES (%s,%s,%s,%s,%s)', (account["Id"],account["username"],_total,_sittostand,result))
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
    cap = cv2.VideoCapture(0)

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
                        print("counter:",counter,"seconds")
                        print("Sit to stand",float("{0:.2f}".format(counter-time_spare)),"seconds")
                        time_spare = counter
                        sit_count.append(0) 
                        print("sit_count",sit_count) 
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
                        print("counter:",counter,"seconds")
                        print("Sit to stand",float("{0:.2f}".format(counter-time_spare)),"seconds")
                        sittostand = counter - time_spare
                        time_spare = counter
                        sit_count.append(0) 
                        print("sit_count",sit_count) 
           

                      
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
    return redirect(url_for('updata'))

  

@app.route("/neverfall")
def neverfall():
    return Response(process_video_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')
    


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
            cursor.execute('INSERT INTO Checklist VALUES (%s,%s, %s, %s, %s, %s, %s, %s)', (id,fall, bendedback, walkslow, cane, medicines, sum, risk))
            cnx.commit()
            msg = 'Thank you for doing the checklist!'
    return render_template("checklist.html", msg = msg)


if __name__ == '__main__':
    app.run(debug=True)