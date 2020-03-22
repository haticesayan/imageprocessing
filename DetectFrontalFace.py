import cv2
import numpy as np

img= cv2.imread('face2.jpg')

#add cascade classifer
yuz_casc = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

#turn image to graylevel
griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect the faces
yuzler = yuz_casc.detectMultiScale(griton,1.1,4)

# give the coordinates and draw rectangle around them
for(x,y,w,h) in yuzler:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

#Show the result image
cv2.imshow('yuzler',img)
cv2.waitKey(0)

#destroy it since job is done
cv2.destroyAllWindows()
