import cv2
import sys

img = cv2.imread("images/soccer.jpg")
if img is None:
	sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
cv2.waitKey(0)
	
