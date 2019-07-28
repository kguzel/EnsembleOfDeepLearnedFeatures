#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:51:02 2019

@author: kadirguzel
"""

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation ,Conv2D , MaxPooling2D ,Flatten
from keras.optimizers import RMSprop
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score


""" 4 boyutlu => n x width x height x 3 """
inputData=np.load("dataSetColorArrayCNN.npy")
outputLabel=np.load("labelArrayColorCNN.npy")


""" Random Split Data-Test 1/7 """
train_img, test_img, train_lbl, test_lbl = train_test_split( inputData, outputLabel, test_size=5/10.0)


""" LabelBinarizer  """
lb = preprocessing.LabelBinarizer()

lb.fit(train_lbl)
train_lbl=lb.transform(train_lbl)

lb.fit(test_lbl)
test_lbl=lb.transform(test_lbl)


""" AlexNet """
model=Sequential();
    
 
# 1st Convolutional Layer
model.add(Conv2D(filters=36, input_shape=(27,27,3), kernel_size=(4,4), strides=(1,1), padding='valid',activation='relu'))


# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1),activation='relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='valid'))


# Max Pooling
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(768,activation='relu'))


# Add Dropout to prevent overfitting
model.add(Dropout(0.2))

# 2nd Fully Connected Layer
model.add(Dense(512))
model.add(Activation('relu'))


# Add Dropout
model.add(Dropout(0.2))

# 3rd Fully Connected Layer
model.add(Dense(4))

model.add(Activation('softmax'))

model.summary()


sgd = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
rmsprop = RMSprop(lr=0.0001, rho=0.6, decay=0.0005)
#adadlt=Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

model.fit(train_img,train_lbl,batch_size=80,epochs=40,validation_split=0.20,shuffle=True)





" TEST"
score=model.evaluate(test_img,test_lbl,verbose = 1)
print('Test Accuracy: ', score[1])


" Test edelim"
y_test_predictions = model.predict(test_img, verbose = 1)


""" LabelBinarizer to decimal """
y_pred_label = np.argmax(y_test_predictions, axis=1)
y_label = np.argmax(test_lbl, axis=1)


" Accuracy"
correct = np.sum(y_pred_label ==  y_label)
print ('Test Accuracy: ', correct/float(test_lbl.shape[0])*100.0, '%')


print('Confusion Matrix')
print(confusion_matrix(y_label, y_pred_label))


fscore = f1_score(y_label, y_pred_label, average=None) 
print('F1 Score')
print(fscore)














