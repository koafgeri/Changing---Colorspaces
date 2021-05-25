import cv2 as cv
import numpy as np

cap = cv.VideoCapture('/home/user/Downloads/LOTR Trailer.mp4')

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("Can't receive frame(stream end).Exiting...")
        break


    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()