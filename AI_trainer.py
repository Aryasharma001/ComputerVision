import cv2
import numpy as np
import math
import time
import pose_estimation_modeule as pm

cap=cv2.VideoCapture("C://Users//ARYA SHARMA//Downloads//pexels-cottonbro-studio-4754028-4096x2160-25fps.mp4")
new_width = 700  # Set the desired width
new_height = 510  # Set the desired height

detector=pm.poseDetector()
count=0
prev_angle2 = 0

while cap.isOpened():
    success,img=cap.read()
    #img=cv2.imread()
    img = cv2.resize(img, (new_width, new_height))
    img=detector.findPose(img,False) 
    lmlist=detector.getPos(img,False)
    #print(lmlist)
    
    
    angle1 = detector.findAngle(img, 24, 26, 28)
    angle2 = detector.findAngle(img, 11, 12, 14)

    if angle2 >= 80 and prev_angle2 < 80:
        count += 1

    prev_angle2 = angle2  # Update prev_angle2 with the current angle2 value

    print(count)
        
        
    
    #cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
    cv2.putText(img,str(count),(15,70),cv2.FONT_HERSHEY_PLAIN,5,(400,0,0),5)
    
    cv2.imshow("Image",img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()   