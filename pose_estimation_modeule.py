import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, upperBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
    
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon =  bool(detectionCon)
        self.trackCon = bool(trackCon)
        self.mpPose=mp.solutions.pose
        self.pose = mp.solutions.pose.Pose(self.upperBody, self.smooth, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        
    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        #print(self.results.pose_landmarks)
            
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=7, circle_radius=5),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=7)
                )
        return img
    def getPos(self,img,draw=True):
        lmlist=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                #print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                 cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmlist         

def main():
    pTime=0
    cap=cv2.VideoCapture("C://Users//ARYA SHARMA//Downloads//pexels-cottonbro-studio-2795750-3840x2160-25fps.mp4")
    frame_height = int(cap.get(4))
    new_width = 700  # Set the desired width
    new_height = 510  # Set the desired height
    
    detector=poseDetector()
    while cap.isOpened():
        success,img=cap.read()
        img=detector.findPose(img)
        lmlist=detector.getPos(img)
        if len(lmlist)!=0:
            print(lmlist[14])
            cv2.circle(img,(lmlist[14][1],lmlist[14][2]),25,(0,0,255),cv2.FILLED)
        
        img = cv2.resize(img, (new_width, new_height)) 
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
if __name__=="__main__":
    main()