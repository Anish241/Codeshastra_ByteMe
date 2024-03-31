import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

mp_hands = mp.solutions.hands

# Load the best model saved during training
best_model = load_model(r'./retrain_best_model.h5')

def get_hand_image(image):
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
                        print("Error: Empty hand image")
                else:
                    print("Error: Invalid hand bounding box")
        else:
            print("Error: No hand detected")
    # Return None if no hand image is found
    return None


def get_pred(sequence):

    labels = ['Correct','Incorrect']
    pred = np.argmax(best_model.predict(np.array(sequence).reshape((1,np.array(sequence).shape[0],np.array(sequence).shape[1],np.array(sequence).shape[2],np.array(sequence).shape[3]))), axis=1)
    # for frame in sequence
    print(labels[pred[0]])
    return sequence[1:], labels[pred[0]]


def write_text_on_frame(frame, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    # Open video capture device (webcam)
    vid = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not vid.isOpened():
        print("Error: Failed to open video capture device")
        return
    
    sequence = []
    while True:

        ret, frame = vid.read()
        hand_box = get_hand_image(frame)

        pred = "" 
        if(len(sequence) < 20 ):
            if(hand_box is not None):
                sequence.append(hand_box)
        else:
            sequence,pred = get_pred(sequence)
        write_text_on_frame(frame, str(pred), position=(50, 50), font_scale=1, color=(255, 255, 255), thickness=2)

        # Display output in OpenCV window
        cv2.imshow('Output', cv2.flip(frame,1))
        if(hand_box is not None):
            sequence.append(hand_box)
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture device
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
