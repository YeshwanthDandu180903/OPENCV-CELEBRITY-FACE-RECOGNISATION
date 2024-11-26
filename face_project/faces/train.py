import numpy as np
import os
import cv2

people=['fahad','nazariya','Rajini_kanth','thalapathy']

haar_face=cv2.CascadeClassifier("D:\\CV\\haar_face.xml")

DIR=r'D:\\CV\\face_project\\faces\\face Recognize'
features,labels=[],[]

def train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            photo=cv2.imread(img_path)
            gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)


            face_detect=haar_face.detectMultiScale(gray,1.3,2)

            for x,y,w,h in face_detect:
                #croppping the image
                face_region=gray[y:y+h,x:x+w]


                features.append(face_region)
                labels.append(label)




train()
print("Training done..................")

#using face_recognizer to train the model using features and labels
face_recognizer=cv2.face.LBPHFaceRecognizer.create()
#we should convert them into arrays the both features and labels
features=np.array(features,dtype='object')
labels=np.array(labels,dtype=int)

face_recognizer.train(features,labels)


#saving the trained
face_recognizer.save('trained_model.yml')

#saving the features and labels using numpy 
np.save("features",features)
np.save("labels",labels)