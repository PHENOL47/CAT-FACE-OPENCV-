#MAJOR PROJECT 2

#NAME - AYUSH KUMAR SHARMA
#YEAR - 2ND  DEPT - ECE
#email - sharmaayushKv@gmail.com
#KALYANI GOVERNMENT ENGINEERING COLLEGE

#USING HAARCASCADE FILE TO IDENTIFY AND TRACK CAT FACE

#OPEN CV PROJECT

#PROGRAM TO IDENTIFY AND TRACK CAT FACE
#VIDEO FILE IS GIVEN IN FOLDER

###########################

import cv2

############################
#Loading the video file and the haarcascade file

cascade_src = r"C:\Users\PHENOL\Desktop\ml projects\haarcascade_frontalcatface.xml"
video_src = r"C:\Users\PHENOL\Desktop\ml projects\cat video.mp4"
objectName = 'CAT'
#############################

cap = cv2.VideoCapture(video_src) #LOADING THE VIDEO 
cat_cascade = cv2.CascadeClassifier(cascade_src) #LOADING THE CASCADE FILE


##############################

while True:

    ret,img = cap.read()
    if(type(img)==type(None)):
        break
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #CONVERTING IT INTO BLACK AND WHITE
    cat = cat_cascade.detectMultiScale(gray , 1.1 , 4) #FOR TRACKIMG THE FACE

    for (x,y,w,h) in cat:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3) #WILL PLACE A RECTANGLE ON CATS FACE
        cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2) #WILL PUT TEXT ON CATS FACE
    cv2.imshow('VIDEO',img)
    if cv2.waitKey(1)==13: #13 ASCII VALUE OF 'ENTER' KEY
        break


cv2.destroyAllWindows()





