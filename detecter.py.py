import cv2
import numpy as np
import serial
usbport = 'COM14'
ser = serial.Serial(usbport, 9600)
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer/trainer.yml')
cascadePath = 'F:\Downloads\Pranjali\opencv\haarcascades\haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(cascadePath);
cam = cv2.VideoCapture(0)
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img =cam.read()
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceDetect.detectMultiScale(gray, 1.3,5)
for(x,y,w,h) in faces:
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
id, conf = recog.predict(gray[y:y+h,x:x+w])
if id==1:
    id="megha"
ser.write('s'.encode())
elif id==2:
    id="pranjali"
    ser.write('s'.encode())
    elif id==3:
    id="annanya"    
    ser.write('s'.encode())
    cv2.putText(img,str(id),(x,y+h),font,0.55,(0,255,0),1)
    cv2.imshow('Face',img)
    if cv2.waitKey(1)==ord('q'):
        break
        cam.release()
        cv2.destroyAllWindows()
