import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np
mp_hands = mp.solutions.hands

def get_hand_image(image):
    if(image is not None):
        width = image.shape[1]
        height = image.shape[0]
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark])
                    hand_bbox = cv2.boundingRect(landmarks.astype(np.float32))
                    # Check if the hand bounding box is valid
                    if hand_bbox[2] > 0 and hand_bbox[3] > 0:
                        # Extract hand image
                        hand_image = image[hand_bbox[1]:hand_bbox[1] + hand_bbox[3], hand_bbox[0]:hand_bbox[0] + hand_bbox[2]]
                        # Check if hand_image is not empty
                        if hand_image.size != 0:
                            # Resize hand image
                            hand_image = cv2.resize(hand_image, (64, 64))
                            return hand_image
                        else:
                            None
                            # print("Error: Empty hand image")
                            
                    else:
                        None
                        # print("Error: Invalid hand bounding box")
                        
            else:
                None
                # print("Error: No hand detected")
                
        # Return None if no hand image is found
        return None

def create_seq_of_frames(vidPath):
    sequence = []
    seq_len = 30
    
    vid = cv2.VideoCapture(vidPath)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("There are {} frames to be processed ".format(total_frames))
    buffer = np.zeros((64, 64,3))
    # Determine the step size to select frames evenly
    step = max(total_frames // seq_len, 1)
    print("Starting video processing")
    for frame_num in range(0, total_frames, step):
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = vid.read()

        hand_frame = get_hand_image(frame)
        # plt.imshow(hand_frame,cmap='gray')

        if hand_frame is None:
            sequence.append(buffer/255)
        else:
            sequence.append(hand_frame / 255)
            buffer = hand_frame
    # Apply pre-padding if necessary
    while len(sequence) < seq_len:
        sequence.append(buffer / 255)
    
    vid.release()
    return sequence[:seq_len] 

def create_dataset(parentFolder, dataset):

    for file in os.listdir(parentFolder):
        filePath = os.path.join(parentFolder, file)  # Use os.path.join to create file paths
        print("Processing " + filePath)
        dataset.append(create_seq_of_frames(filePath))
        print("Video processed")
    return np.array(dataset)  # Convert the list of sequences into a numpy array

def prepare(path):
    x = []
    x = create_dataset(path, x)  # Call create_dataset to prepare the dataset
    # print(x.shape)
    return x

def prediction(input):
    from keras.models import load_model

    # Load the best model saved during training
    model = load_model('30frame_best_model.h5')

    return np.argmax(model.predict(input), axis=1)