
import cv2
import mediapipe as mp
import time

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while cap.isOpened():
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results)
    
    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            for id,lm in enumerate(handLMs.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                
                cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLMs,mpHands.HAND_CONNECTIONS)
            
    cTime=time.time()
    fps=1/(cTime-pTime)  
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)      
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    
