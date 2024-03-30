import cv2
import numpy as np
from keras.models import load_model

# Load the best model saved during training
best_model = load_model(r'./retrain_best_model.h5')

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
        pred = "" 
        if(len(sequence) < 20 ):
            sequence.append(cv2.resize(frame,(64,64)))
        else:
            sequence,pred = get_pred(sequence)
        write_text_on_frame(frame, str(pred), position=(50, 50), font_scale=1, color=(255, 255, 255), thickness=2)

        # Display output in OpenCV window
        cv2.imshow('Output', cv2.flip(frame,1))
        sequence.append(cv2.resize(frame,(64,64)))
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture device
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
