import cv2
import sys
import os

dirname = os.path.dirname(__file__)

cap = cv2.VideoCapture(dirname + "videos/overpass.mp4")
while (cap.isOpened()):
	ret, frame = cap.read()
	cv2.imshow("Display window", frame)
	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()
