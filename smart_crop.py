import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class smart_crop:
    def __init__(self, img,condition):
        self.image = img
        self.condition= condition

    def __contour_arr(self):
        import numpy as np
        import cv2

        # Read the input image
        arr = []
        img = cv2.imread(self.image)
        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply thresholding to convert the grayscale image to a binary image
        ret, thresh = cv2.threshold(gray, 100, 255, 0)
        if(self.condition == True):
            thresh =cv2.bitwise_not(thresh)


        # find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of contours detected:", len(contours))
        for i in range(0, len(contours)):
            # take first contour
            cnt = contours[i]

            # define the precision
            epsilon = 0.01 * cv2.arcLength(cnt, True)

            # approximate the contour
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for i in range(0, len(approx)):
                arr.append(approx[i])
            # array.append(approx)

            # draw the contour on the input image
            cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)

            # draw the approximate contour on the input image
            cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
        return arr

        # display the image with drawn contours and approximate contours

    def __x_y_points(self, array):
        x_points = []
        y_points = []
        for i in range(0, len(array)):
            x_points.append(array[i][0][0])
            y_points.append(array[i][0][1])

        return x_points, y_points

    def __x_y_min_max(self, x_points, y_points):

        x_max, y_max = max(x_points), max(y_points)
        x_min, y_min = min(x_points), min(y_points)
        return x_max, y_max, x_min, y_min

    def crop(self):
        from PIL import Image
        img = cv2.imread(self.image)
        array = self.__contour_arr()

        x_points, y_points = self.__x_y_points(array)
        xmax, ymax, xmin, ymin = self.__x_y_min_max(x_points, y_points)
        # cv2.rectangle(img,(x_max+2,y_min-2),(x_min-2,y_max+3),(36,255,12), 1)
        # Setting the points for cropped image
        im = img[ymin-2:ymax+3,xmin-2:xmax+3 ]
        #im = Image.open(self.image)
        left = xmin
        top = ymin
        right = xmax
        bottom = ymax
        if (self.condition == True):
            im = cv2.bitwise_not(im)
        # Cropped image of above dimension
        # (It will not change original image)
        # box=(left, upper, right, lower)
        #im1 = im.crop((left - 2, top - 2, right + 2, bottom + 2))

        return im
        # Shows the image in image viewer

#letter = smart_crop(r"C:\Users\ARAS STORE\Desktop\3.jpg")
#plt.imshow(letter.crop(),cmap='gray')
#plt.show()
