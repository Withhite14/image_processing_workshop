import cv2
import sys
import os

dirname = os.path.dirname(__file__)

img = cv2.imread(dirname + "images/soccer.jpg")
if img is None:
	sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
cv2.waitKey(0)
	
