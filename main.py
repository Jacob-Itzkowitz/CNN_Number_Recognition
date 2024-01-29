from PIL import Image
import numpy as np
import cv2

#opens image and saves it to a variable
Zero = cv2.imread(r"C:\Users\jacob.itzkowitz\PycharmProjects\ImageRecognitionNumbers\Images\0.jpg")

#gray scale image to reduce noise
grayscale_image = cv2.cvtColor(Zero, cv2.COLOR_BGR2GRAY)

#finding edges
edges = cv2.Canny(grayscale_image, 50, 150, apertureSize=3)

Lines_list = []
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=5, maxLineGap=10)

for points in lines:
    x, y, x1, y1 = points[0]
    cv2.line(Zero, (x, y), (x1, y1), (0, 255, 0), 2)
    Lines_list.append([(x, y), (x1, y1)])


cv2.imwrite("Lines.png", Zero)



