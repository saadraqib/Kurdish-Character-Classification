import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from TrainingTheModel import *
from smart_crop import *

class Img_preprocessing:
    def __init__(self):
        pass

    def __noise_removal(self, image):
        import cv2
        #img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(image, kernel, iterations=1)
        img = cv2.erode(image, kernel, iterations=1)
        img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        img = cv2.medianBlur(image, 3)
        return img

    def binarize(self, image):
        img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        no_noise = self.__noise_removal(img_array)
        im_RGB = cv2.cvtColor(no_noise, cv2.COLOR_BGR2RGB)
        im_gray = cv2.cvtColor(im_RGB, cv2.COLOR_RGB2GRAY)

        th, im_gray = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
        inverted_img = cv2.bitwise_not(im_gray)
        return inverted_img

    def image2array(self, path, pixel=28):
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        from PIL import Image
        img_pil = Image.fromarray(img_array)
        img_28x28 = np.array(img_pil.resize((pixel, pixel), Image.ANTIALIAS))

        img_array = (img_28x28.flatten())

        img_array = img_array.reshape(-1, 1).T

        return img_array

img = r"C:\Users\ARAS STORE\Desktop\DataSet\15.jpg"

print(img[-3:])
process = Img_preprocessing()
image = process.binarize(img)
path = r"C:\Users\ARAS STORE\Desktop\DataSet\data_no_preprocessed\{}.jpg".format(img[-3:])


letter = smart_crop(img,True)
croped_image = letter.crop()
cv2.imwrite(path,croped_image)
plt.imshow(croped_image,cmap='gray')
plt.show()

img_arr = process.image2array(path,28)

data = pd.read_csv(r"C:\Users\ARAS STORE\Desktop\DataSet\kurdish_alphabetical.csv")
model = SVC_Algorithm(data)
model.splitting_x_y()
model.feature_scaling()
cls = model.svc_classifier()
#xtest = model.feature_scaling()[1]

ypred = cls.predict(img_arr)
myList = ["ئ","د","ف","ا","ق","ڤ","ک","گ","ل","ڵ","م","ن","ر","و","ۆ","ب","وو","ه","ە","ی","ێ","ت","پ","ڕ","ج","چ","ح","خ","ك","ز","ژ","س","ش","ع","غ",]
print(ypred)
print(myList[ypred[0]])