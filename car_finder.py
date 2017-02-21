import os, sys

import cv2

# Select a video & a cascade
video_source = 'mclaren.avi'
cascade_source = 'cars.xml'


#set avi source as capture source, convert cascade xml to opencv cascade
cap = cv2.VideoCapture(video_source)
car_cascade = cv2.CascadeClassifier(cascade_source)



while True:
    ret, img = cap.read() #grabs a frame from the video
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale

    cars = car_cascade.detectMultiScale(gray, 1.5, 1) #does the cascadin'

    for (x,y,w,h) in cars:  #draw rectangles
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,126,255),2)

    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()



cv2.waitKey(0)
