
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class SVC_Algorithm:
    def __init__(self, dataset):
        import numpy as np
        self.dataset = dataset


    def x_values(self, i=0,j=1):
        x_values = self.dataset.iloc[i:,j:]
        return x_values

    def y_values(self, i=0):
        y_values = self.dataset[f"{i}"]
        return y_values

    def y_valuesToLetter(self,dic):
        pass

    def splitting_x_y(self, tsize=0.33, rstate=0):
        from sklearn.model_selection import train_test_split
        x = self.x_values()
        y = self.y_values()
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=tsize, random_state=rstate)
        return xtrain, xtest, ytrain, ytest

    def feature_scaling(self):
        from sklearn.preprocessing import StandardScaler
        stX = StandardScaler()
        xtrain = self.splitting_x_y()[0]
        xtest = self.splitting_x_y()[1]
        xtrain = stX.fit_transform(xtrain)
        xtest = stX.transform(xtest)
        return xtrain, xtest

    def svc_classifier(self, kernel='linear', rstate=0):
        from sklearn.svm import SVC
        classifier = SVC(kernel=kernel, random_state=rstate)
        # classifier.fit(xtrain,ytrain)
        xtrain = self.feature_scaling()[0]
        ytrain = self.splitting_x_y()[2]
        classifier.fit(xtrain, ytrain)
        return classifier



#data = pd.read_csv(r"C:\Users\ARAS STORE\Desktop\DataSet\kurdish_alphabetical.csv")
#model = SVC_Algorithm(data)
#model.splitting_x_y()
#model.feature_scaling()
#cls = model.svc_classifier()
#xtest = model.feature_scaling()[1]

#ypred = cls.predict(xtest[7].reshape(-1,1).T)
#print(ypred)


