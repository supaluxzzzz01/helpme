import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Initiate a timer
start_time = None
end_time = None
prev_time = 0
timer_started = None
countdown = 5
# Initiate counters
sits = 0
stage = None

# Set up Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
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
        
        start_countdown = False
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            continue
        
        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract coordinate values of interest
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y
        
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        
        # Calculate hip-knee ratio
        #hip_knee_ratio = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y - left_knee_y) / (right_shoulder_y - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
        
        # Stage 1: Check if right hand is above right shoulder
        if stage == None:
            if right_hand_y < right_shoulder_y and left_hand_y < left_shoulder_y:
                curr_time = time.time()
                start_countdown = True
                stage = 1
                start_time = time.time() + 5
                prev_time = start_time
                timer_started = True 
                text = "Timer Started"   
            else:
                text = "Raise both hand above shoulder to start timer"
        # Stage 2: Detecting sits and measuring time
        elif stage == 1 and start_time >= 0:
            if knee_y - hip_y < 0.03:
                sits += 1
                stage = 2
                end_time = time.time()
                text = "Sit Detected, Total Sits: " + str(sits) + ", Time: " + str(round(end_time - start_time, 2)) + "s"
            else:
                text = "Stand Straight"
                
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
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
