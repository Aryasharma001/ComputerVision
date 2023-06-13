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



class handDetector():
    def __init__(self, mode=False, max_Hands=2, detection_confidence=0.5, model_complexity=0, track_confidence=0.5):
        self.mode = mode
        self.max_Hands = max_Hands
        self.detection_confidence = detection_confidence
        self.modelComplex = 1
        self.track_confidence = track_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_Hands, self.modelComplex, self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)
        
        return img
    def findPosition(self,img,handNo=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                    #print(id,lm)
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
            
                    lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)
                    
                    
        return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while cap.isOpened():
        success, img = cap.read()
        img = detector.find_hands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist)!=0:
         print(lmlist[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
