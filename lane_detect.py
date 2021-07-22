import cv2
import numpy as np

def region_of_interest(img, vertices):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, np.int32([shape]), match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

cap  = cv2.VideoCapture('/home/skuba/image_processing_workshop/videos/lane_detect.mp4')

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0.5, height),
        (width/2, height/2),
        (width, height)
    ]
    
    masked = color_filter(image)
    gray_image = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)
    try:
        lines = cv2.HoughLinesP(cropped_image,
                                rho=6,
                                theta=np.pi/180,
                                threshold=60,
                                lines=np.array([]),
                                minLineLength=20,
                                maxLineGap=80)
        image_with_lines = drow_the_lines(image, lines)
    except:
        pass
    cv2.imshow("color_filter", masked)
    cv2.imshow("gray", gray_image)
    cv2.imshow("edge", canny_image)
    cv2.imshow("ROI", cropped_image)
    cv2.imshow("Line", image_with_lines )
    #cv2.imshow("Display window", image_with_lines)
    if cv2.waitKey(1) == 27:
        break