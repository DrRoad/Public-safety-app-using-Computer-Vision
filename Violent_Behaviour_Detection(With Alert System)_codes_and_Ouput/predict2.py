import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
from trip_alert2 import alert

# The file containing the frozen graph of the trained model
MODEL_FILE = 'pb/individual_based_on_images.pb'

IMAGE_SIZE = 120
NUM_CHANNELS = 1
BEHAVIOUR=['Abnormal','Normal']
COLOR = [(0,0,255),(255,0,0)]
unusual=0

cap=cv2.VideoCapture('1videoplayback')
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(MODEL_FILE,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Get input and output tensors from graph
        x_input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")
        # Read a frame
        success, frame = cap.read()
        while success:
            

            # Convert it to grayscale for recognition
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY )


            # Crop out the face and do necessary preprocessing
            gray=cv2.resize(gray,(IMAGE_SIZE,IMAGE_SIZE)).reshape((1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
            gray=(gray-np.mean(gray))*(2.0/255.0)

            # Feed the cropped and preprocessed frame to classifier
            result=sess.run(output,{x_input:gray})

            # Get the behaviour and confidence
            action=BEHAVIOUR[np.argmax(result)]
            if action=='Abnormal':
            	unusual+=1
            	#if unusual%100==1:
            		#alert()
            disp_color=COLOR[np.argmax(result)]
            confidence=np.max(result)
            print(action)
            #print('argmax(result)',np.argmax(result))
            cv2.putText(frame,action,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,disp_color,2,cv2.LINE_AA)
            

            """
            # Draw rectangle and write text on frame to be displayed
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,emotion+": %.1f"%confidence,(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Image Fed to Classifier', cv2.resize(gray.reshape((48,48)), (48,48)))
            """
            cv2.imshow('Action', cv2.resize(frame, (800,650)))
            
            # Read a frame
            success, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
