import cv2
import numpy as np
import pafy

#USA LIVE CAM
#url = 'https://www.youtube.com/watch?v=5_XSYlAfJZM'

#JAPAN LIVE CAM
url = 'https://www.youtube.com/watch?v=RQA5RcIZlAM'

video = pafy.new(url)
best = video.getbest(preftype="mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

cap = cv2.VideoCapture(best.url)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    mask = object_detector.apply(frame1)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    kernel=np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (area < 3500):
            continue
        # if (area < 4000):
        #     cv2.putText(frame1, 'Person', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        # else: cv2.putText(frame1, 'Vehicle', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    cv2.imshow("feed", frame1)
    #cv2.imshow("mask", mask)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(2) == 27:
        break

cv2.destroyAllWindows()
cap.release()