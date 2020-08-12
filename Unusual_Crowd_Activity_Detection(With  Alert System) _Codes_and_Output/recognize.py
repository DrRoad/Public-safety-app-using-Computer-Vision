"""
Title: Emotion Recognition (Happy / Sad / Neutral)
Author: Sayali Deshpande
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
from trip_alert2 import alert

# The file containing the frozen graph of the trained model
MODEL_FILE = 'pb/crowd_frame_final.pb'

IMAGE_SIZE = 120
NUM_CHANNELS = 1
ACTIVITY=['Unusual Crowd Activity','Normal']
COLOR=[(0,0,255),(255,0,0)]
HEIGHT=300
WIDTH=400
VIDEO_FILE='FinalVid/Abnormal/fans_violence__Hardcore_Supporter_Fight_Extreme_Violence_Final_Four_Volley_Praha__javierulf__jzmiAbjN6Mw.avi'
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

cap=cv2.VideoCapture(VIDEO_FILE)
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

        # Take first frame and find corners in it
        success, old_frame = cap.read()
        old_frame=cv2.resize(old_frame,(WIDTH,HEIGHT))
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        count=0
        unusual=0
        # Read a frame
        success, frame = cap.read()

        while success:

            # Convert it to grayscale for recognition
            frame=cv2.resize(frame,(WIDTH,HEIGHT))
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY )


            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

            if p1 is None:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)


            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i%100].tolist(), 2)
                frame  = cv2.circle(frame,(a,b),5,color[i%100].tolist(),-1)
            img = cv2.add(frame,mask)


            # Now update the previous frame and previous points
            old_gray = gray.copy()
            p0 = good_new.reshape(-1,1,2)
            
            
            count+=1
            if count%25==0:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                mask = np.zeros_like(old_frame)
            """
            elif count%25==0:
                ptemp=cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                p0=np.append(p0,ptemp,axis=0)
            """
            toClassify=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY ).astype(float)
            toClassify=cv2.resize(toClassify,(IMAGE_SIZE,IMAGE_SIZE)).reshape((1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
            toClassify=(toClassify-np.mean(toClassify))*(1.0/255.0)


            # Feed the cropped and preprocessed frame to classifier
            result=sess.run(output,{x_input:toClassify})

            # Get the emotion and confidence
            crowd=ACTIVITY[np.argmax(result)]
            if crowd=='Unusual Crowd Activity':
                unusual+=1
                if (unusual+10)%100==1:
                    ##
            disp_color=COLOR[np.argmax(result)]
            confidence=np.max(result)    
            print(crowd)
            cv2.putText(img,crowd,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,disp_color,2,cv2.LINE_AA)
            cv2.imshow('frame',cv2.resize(img,(WIDTH*2,HEIGHT*2)))
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            # Read a frame
            success, frame = cap.read()

           

            cv2.imshow('Image Fed to Classifier', toClassify.reshape(IMAGE_SIZE,IMAGE_SIZE))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

