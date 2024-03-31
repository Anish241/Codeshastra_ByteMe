import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3
import threading
import time
import matplotlib.pyplot as plt

def calculate_angle(landmarks, joints):
    a = np.array(landmarks[joints[0]])
    b = np.array(landmarks[joints[1]])
    c = np.array(landmarks[joints[2]])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

shaking_treshold = 10
prev_landmark = None
start_time = time.time()
shaking_frequency = 0
total_shakes = 0

freqs = []

def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()
def async_speak(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands, mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, mp_pose.Pose(min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Hand detections
        hand_results = hands.process(image)
        
        # Face detections
        face_results = face_detection.process(image)

        # pose detection
        pose_results = pose.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if pose_results.pose_landmarks:
            landmarks = np.array([[lm.x * width, lm.y * height] for lm in pose_results.pose_landmarks.landmark])
            # angle_r_shoulder_elbow_wrist = calculate_angle(landmarks, [mp_pose.PoseLandmark.RIGHT_SHOULDER, 
            #                                                            mp_pose.PoseLandmark.RIGHT_ELBOW, 
            #                                                            mp_pose.PoseLandmark.RIGHT_WRIST])
            # if angle_r_shoulder_elbow_wrist < 90:
            #     async_speak("Right arm is bent")
            
            # Calculate angle between shoulder, hip, and knee for back straightness
            angle_r_shoulder_hip_knee = calculate_angle(landmarks, [mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                                                    mp_pose.PoseLandmark.RIGHT_HIP, 
                     
                                                                    mp_pose.PoseLandmark.RIGHT_KNEE])
            print(angle_r_shoulder_hip_knee)
            if angle_r_shoulder_hip_knee < 160:
                async_speak("Back is not straight")
            if  angle_r_shoulder_hip_knee > 160:   
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        landmarks = np.array([[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark])
                        hand_bbox = cv2.boundingRect(landmarks.astype(np.float32))
                        
                        # Draw bounding box around hand
                        cv2.rectangle(image, (int(hand_bbox[0]), int(hand_bbox[1])), 
                                    (int(hand_bbox[0]+hand_bbox[2]), int(hand_bbox[1]+hand_bbox[3])), 
                                    (0, 255, 0), 2)
                        
                        # Extract tip of the hand (assuming it's at the bottom when held vertically)
                        tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        
                        # Convert tip landmark to image coordinates
                        tip_x = int(tip_landmark.x * width)
                        tip_y = int(tip_landmark.y * height)
                        
                        # Hand position
                        hand_position = (tip_x, tip_y)
                        
                        # Face detections
                        if face_results.detections:
                            for detection in face_results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                
                                # Convert normalized bounding box coordinates to pixels
                                ih, iw, _ = image.shape
                                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                    int(bboxC.width * iw), int(bboxC.height * ih)
                                
                                # Draw bounding box around face
                                cv2.rectangle(image, bbox, (255, 0, 0), 2)
                                
                                # Face position (considering the center of the bounding box)
                                face_position = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                                
                                # Calculate distance between hand tip and face position
                                distance = math.sqrt((hand_position[0] - face_position[0])**2 + (hand_position[1] - face_position[1])**2)
                                
                                # Display distance on the image
                                # cv2.putText(image, f"Distance: {distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                # Calculate wrist and finger angles only if distance is between 100 to 150 pixels
                                if  distance <= 250:
                                    cv2.putText(image, f"Distance: OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    # Check if wrist is straight with vertical
                                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                                    angle_with_vertical = math.degrees(math.atan2(wrist.y - middle_finger_tip.y, wrist.x - middle_finger_tip.x))
                                    # normalize the angle to be between 0 and 180 degrees and check if it's within 85 to 105 degrees
                                    angle_with_vertical = abs(angle_with_vertical) % 180
                                    # Proceed with angle calculation only if wrist is straight with vertical
                                    if  48 <= angle_with_vertical <= 105:
                                        # Check which finger is at the top
                                        # cv2.putText(image, f"Angle : {angle_with_vertical:.2f} degrees", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                                        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                                        # check which finger is at the top and set the top finger
                                        top_finger = None
                                        top_tip = None
                                        if thumb_tip.y < index_finger_tip.y and thumb_tip.y < middle_finger_tip.y and thumb_tip.y < ring_finger_tip.y and thumb_tip.y < pinky_tip.y:
                                            top_finger = "Thumb"
                                            top_tip = thumb_tip
                                        elif index_finger_tip.y < thumb_tip.y and index_finger_tip.y < middle_finger_tip.y and index_finger_tip.y < ring_finger_tip.y and index_finger_tip.y < pinky_tip.y:
                                            top_finger = "Index Finger"
                                            top_tip = index_finger_tip
                                        elif middle_finger_tip.y < thumb_tip.y and middle_finger_tip.y < index_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y and middle_finger_tip.y < pinky_tip.y:
                                            top_finger = "Middle Finger"
                                            top_tip = middle_finger_tip
                                        elif ring_finger_tip.y < thumb_tip.y and ring_finger_tip.y < index_finger_tip.y and ring_finger_tip.y < middle_finger_tip.y and ring_finger_tip.y < pinky_tip.y:
                                            top_finger = "Ring Finger"
                                            top_tip = ring_finger_tip
                                        else:
                                            top_finger = "Pinky"
                                            top_tip = pinky_tip
                                        
                                        base_tip = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                        # Calculate angle with the top finger
                                        angle = math.degrees(math.atan2(top_tip.y - base_tip.y , top_tip.x - base_tip.x))

                                        # Draw angles on the image
                                        if top_finger == "Thumb":
                                            print(angle)
                                            if abs(angle) > 106.5:
                                                async_speak("Thumb bent")
                                        elif top_finger == "Index Finger":
                                            if abs(angle) >106.5:
                                                async_speak("Index Finger bent")
                                        else:
                                            async_speak("Please use only Thumb or Index Finger")
                                            print("Please use only Thumb or Index Finger")
                                        if prev_landmark is not None:
                                            disp = np.linalg.norm(landmarks-prev_landmark)
                                            if disp > shaking_treshold:
                                                total_shakes += 1
                                                shaking_frequency = total_shakes / (time.time() - start_time)
                                                freqs.append(shaking_frequency)
                                                # cv2.putText(image, f"Shaking Freq: {shaking_frequency},", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        prev_landmark = landmarks.copy()
                                    else:
                                        async_speak("Please keep your wrist straight with vertical")
                                        print("Please keep your wrist straight with vertical")
                                else:
                                    cv2.putText(image, f"Distance: Too far", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    print("Please keep your hand closer to the face")
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# plot the shaking frequency over time
plt.plot(freqs)
plt.xlabel("Time (s)")
plt.ylabel("Total Shakes")
plt.title("Total Shakes over time")
plt.show()

