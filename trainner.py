import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path="dataSet"

##detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        print(ID) 
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
        
    return IDs,faces

Ids,faces=getImageWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.write('recognizer/trainingData.yml')
cv2.destroyAllWindows()

##
##
##def getImagesAndLabels(path):
##    #get the path of all the files in the folder
##    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
##    #create empth face list
##    faceSamples=[]
##    #create empty ID list
##    Ids=[]
##    #now looping through all the image paths and loading the Ids and the images
##    for imagePath in imagePaths:
##        #loading the image and converting it to gray scale
##        pilImage=Image.open(imagePath).convert('L')
##        #Now we are converting the PIL image into numpy array
##        imageNp=np.array(pilImage,'uint8')
##        #getting the Id from the image
##        Id=int(os.path.split(imagePath)[-1].split(".")[1])
##        # extract the face from the training image sample
##        faces=detector.detectMultiScale(imageNp)
##        #If a face is there then append that in the list as well as Id of it
##        for (x,y,w,h) in faces:
##            faceSamples.append(imageNp[y:y+h,x:x+w])
##            Ids.append(Id)
##    return faceSamples,Ids
##
##
##faces,Ids = getImagesAndLabels('dataSet')
##recognizer.train(faces, np.array(Ids))
##recognizer.save('trainner/trainner.yml')

