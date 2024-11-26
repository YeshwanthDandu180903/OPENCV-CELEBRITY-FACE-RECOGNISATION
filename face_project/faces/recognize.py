import cv2
import numpy as np


input_image=cv2.imread("D:\\CV\\face_project\\faces\\face_validation\\thalapathy\\t2.jpeg")
#resizingg the image
def resize_image(input_image,scale=200):
    height=input_image.shape[0]*scale//100
    width=input_image.shape[1]*scale//100

    dimensions=(width,height)
    return cv2.resize(input_image,dimensions,cv2.INTER_LINEAR)
people=['fahad','nazariya','Rajini_kanth','thalapathy']

haar_face=cv2.CascadeClassifier("D:\\CV\\haar_face.xml")
if haar_face.empty():
    raise IOError("Failed to load haar_face.xml")

resized=resize_image(input_image)

#now just loading the features and labels using numpy
features=np.load('D:\\CV\\face_project\\features.npy',allow_pickle=True)
labels=np.load('D:\\CV\\face_project\\labels.npy',allow_pickle=True)


#using face recognizer to read 
face_recognizer=cv2.face.LBPHFaceRecognizer.create()

face_recognizer.read('D:\\CV\\face_project\\trained_model.yml')

#converting the image into gray for procesing the haar_face

gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray",gray)

face_detect=haar_face.detectMultiScale(gray,1.3,2)
if len(face_detect) == 0:
    print("No faces detected in the image.")
else:
    for x,y,w,h in face_detect:
        face_region=gray[y:y+h,x:x+w]

        #now using face_recognizer.predict
        label,confidence=face_recognizer.predict(face_region)
        print(people[label])
        print(f" Person name is {people[label]} confidence level is {confidence}")


        #putting the text on the image
        cv2.putText(resized,str(people[label]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.4,(0,255,0),3)
        face_rect=cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),3)


#displaying the image

cv2.imshow("Deteced Face",resized)
cv2.waitKey(0)
